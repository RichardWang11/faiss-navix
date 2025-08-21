#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <iomanip>
#include <sstream>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// 详细性能分析结构
struct DetailedTiming {
    std::chrono::microseconds preparation_time;
    std::chrono::microseconds search_core_time;  
    std::chrono::microseconds filtering_time;
    std::chrono::microseconds postprocess_time;
    std::chrono::microseconds total_time;
    
    // 搜索统计信息
    size_t nodes_visited;
    size_t distance_computations;
    size_t candidates_evaluated;
    size_t filtered_out;
};

// Helper function: Outputs to both console and a log file
template<typename T>
void log_output(std::ofstream& logfile, const T& content) {
    std::cout << content;
    logfile << content;
}

// Utility function to read .fvecs files (float vectors)
std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << " for reading.\n";
        return {};
    }
    
    std::vector<std::vector<float>> dataset;
    while (file) {
        int d;
        if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;
        
        std::vector<float> vec(d);
        if (!file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float))) break;
        
        dataset.push_back(std::move(vec));
    }
    file.close();
    
    std::cout << "Loaded " << dataset.size() << " vectors of dimension " 
              << (dataset.empty() ? 0 : dataset[0].size()) << " from " << filename << std::endl;
    
    return dataset;
}

// Utility function to read .ivecs files (integer vectors, for ground truth)
std::vector<std::vector<int>> read_ivecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << " for reading.\n";
        return {};
    }

    std::vector<std::vector<int>> dataset;
    while (file) {
        int d;
        if (!file.read(reinterpret_cast<char*>(&d), sizeof(int))) break;

        std::vector<int> vec(d);
        if (!file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(int))) break;

        dataset.push_back(std::move(vec));
    }
    file.close();

    std::cout << "Loaded " << dataset.size() << " ground truth sets of size "
              << (dataset.empty() ? 0 : dataset[0].size()) << " from " << filename << std::endl;

    return dataset;
}

// Reads numeric labels from a .jsonl file
std::vector<int> read_jsonl_labels(const std::string& filename, const std::string& attribute_name) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error: Unable to open file " << filename << " for reading.\n";
        return {};
    }
    
    std::vector<int> labels;
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        try {
            json j = json::parse(line);
            if (j.contains(attribute_name)) {
                int label = j[attribute_name].get<int>();
                labels.push_back(label);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing JSON line: " << e.what() << std::endl;
        }
    }
    
    std::cout << "Loaded " << labels.size() << " labels from " << filename << std::endl;
    return labels;
}

// Converts a vector of vectors into a flat float array
std::vector<float> flatten_vectors(const std::vector<std::vector<float>>& vectors) {
    if (vectors.empty()) return {};
    
    int dimension = vectors[0].size();
    std::vector<float> flattened(vectors.size() * dimension);
    
    for (size_t i = 0; i < vectors.size(); ++i) {
        std::copy(vectors[i].begin(), vectors[i].end(), 
                 flattened.begin() + i * dimension);
    }
    
    return flattened;
}

// 带性能分析的IDSelector
class ProfilingLabelSelector : public faiss::IDSelector {
private:
    const std::vector<int>& db_labels_;
    int target_label_;
    mutable size_t filter_calls_;
    mutable size_t filtered_out_count_;
    mutable std::chrono::microseconds filter_time_;
    
public:
    ProfilingLabelSelector(const std::vector<int>& db_labels, int target_label) 
        : db_labels_(db_labels), target_label_(target_label), 
          filter_calls_(0), filtered_out_count_(0), filter_time_(0) {}
    
    bool is_member(faiss::idx_t id) const override {
        auto start = std::chrono::high_resolution_clock::now();
        
        filter_calls_++;
        bool result = false;
        
        if (id >= 0 && id < db_labels_.size()) {
            result = (db_labels_[id] == target_label_);
        }
        
        if (!result) {
            filtered_out_count_++;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        filter_time_ += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return result;
    }
    
    void reset_stats() {
        filter_calls_ = 0;
        filtered_out_count_ = 0;
        filter_time_ = std::chrono::microseconds(0);
    }
    
    size_t get_filter_calls() const { return filter_calls_; }
    size_t get_filtered_out_count() const { return filtered_out_count_; }
    std::chrono::microseconds get_filter_time() const { return filter_time_; }
};

// Calculates recall for a single query
float calculate_recall(const std::vector<int>& ground_truth, const std::vector<faiss::idx_t>& results) {
    if (ground_truth.empty()) {
        return 1.0f;
    }
    std::unordered_set<int> ground_truth_set(ground_truth.begin(), ground_truth.end());
    int match_count = 0;
    for (faiss::idx_t result_id : results) {
        if (ground_truth_set.count(result_id)) {
            match_count++;
        }
    }
    return static_cast<float>(match_count) / ground_truth.size();
}

// Navix详细性能分析
DetailedTiming profile_navix_search(
    faiss::IndexHNSWFlat& index,
    const float* query,
    int k,
    int dimension,
    const std::vector<int>& db_labels,
    int query_label,
    std::vector<faiss::idx_t>& labels,
    std::vector<float>& distances) {
    
    DetailedTiming timing;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 1. 准备阶段：创建filter mask
    auto prep_start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> filter_mask(db_labels.size());
    size_t eligible_count = 0;
    for (size_t i = 0; i < db_labels.size(); ++i) {
        if (db_labels[i] == query_label) {
            filter_mask[i] = 1;
            eligible_count++;
        } else {
            filter_mask[i] = 0;
        }
    }
    faiss::VisitedTable vt(db_labels.size());
    faiss::HNSWStats stats;
    auto prep_end = std::chrono::high_resolution_clock::now();
    timing.preparation_time = std::chrono::duration_cast<std::chrono::microseconds>(prep_end - prep_start);
    
    // 2. 核心搜索阶段
    auto search_start = std::chrono::high_resolution_clock::now();
    index.navix_single_search(
        query, k, distances.data(), labels.data(),
        reinterpret_cast<char*>(filter_mask.data()), vt, stats);
    auto search_end = std::chrono::high_resolution_clock::now();
    timing.search_core_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start);
    
    // 3. 过滤时间（在navix中集成在搜索过程中）
    timing.filtering_time = std::chrono::microseconds(0); // Navix中过滤是集成的
    
    // 4. 后处理时间（minimal for navix）
    auto post_start = std::chrono::high_resolution_clock::now();
    // 这里可以添加任何后处理逻辑
    auto post_end = std::chrono::high_resolution_clock::now();
    timing.postprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start);
    
    auto total_end = std::chrono::high_resolution_clock::now();
    timing.total_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    // 收集统计信息
    timing.nodes_visited = stats.n1 + stats.n2;
    timing.distance_computations = stats.ndis;
    timing.candidates_evaluated = eligible_count;
    timing.filtered_out = db_labels.size() - eligible_count;
    
    return timing;
}

// HNSW+Filter详细性能分析
DetailedTiming profile_hnsw_filter_search(
    faiss::IndexHNSWFlat& index,
    const float* query,
    int k,
    int dimension,
    const std::vector<int>& db_labels,
    int query_label,
    std::vector<faiss::idx_t>& labels,
    std::vector<float>& distances) {
    
    DetailedTiming timing;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 1. 准备阶段：创建selector
    auto prep_start = std::chrono::high_resolution_clock::now();
    ProfilingLabelSelector selector(db_labels, query_label);
    faiss::SearchParametersHNSW search_params;
    search_params.sel = &selector;
    search_params.efSearch = 200;
    auto prep_end = std::chrono::high_resolution_clock::now();
    timing.preparation_time = std::chrono::duration_cast<std::chrono::microseconds>(prep_end - prep_start);
    
    // 2. 搜索阶段（包含过滤）
    auto search_start = std::chrono::high_resolution_clock::now();
    index.search(1, query, k, distances.data(), labels.data(), &search_params);
    auto search_end = std::chrono::high_resolution_clock::now();
    timing.search_core_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start);
    
    // 3. 过滤时间（从selector获取）
    timing.filtering_time = selector.get_filter_time();
    
    // 4. 后处理时间
    auto post_start = std::chrono::high_resolution_clock::now();
    // 后处理逻辑
    auto post_end = std::chrono::high_resolution_clock::now();
    timing.postprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start);
    
    auto total_end = std::chrono::high_resolution_clock::now();
    timing.total_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
    
    // 收集统计信息
    timing.nodes_visited = 0; // 标准HNSW不直接提供这个统计
    timing.distance_computations = 0;
    timing.candidates_evaluated = selector.get_filter_calls();
    timing.filtered_out = selector.get_filtered_out_count();
    
    return timing;
}

int main() {
    // --- File Setup ---
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    std::stringstream log_ss;
    log_ss << "navix_profiling_log_" 
           << std::put_time(&tm, "%Y%m%d_%H%M%S") 
           << ".txt";
    
    std::ofstream logfile(log_ss.str());
    if (!logfile.is_open()) {
        std::cerr << "Error: Cannot create log file " << log_ss.str() << std::endl;
        return -1;
    }
    std::cout << "Log file created: " << log_ss.str() << std::endl;

    std::stringstream csv_ss;
    csv_ss << "navix_profiling_results_"
           << std::put_time(&tm, "%Y%m%d_%H%M%S")
           << ".csv";
    std::ofstream csvfile(csv_ss.str());
    if (!csvfile.is_open()) {
        std::cerr << "Error: Cannot create CSV file " << csv_ss.str() << std::endl;
        return -1;
    }

    // 写入CSV头部
    csvfile << "Query,Label,Selectivity,"
            << "Navix_Total,Navix_Prep,Navix_Search,Navix_Filter,Navix_Post,"
            << "HNSW_Total,HNSW_Prep,HNSW_Search,HNSW_Filter,HNSW_Post,"
            << "Navix_NodesVisited,Navix_DistComps,Navix_Candidates,Navix_FilteredOut,"
            << "HNSW_Candidates,HNSW_FilteredOut,Navix_Recall,HNSW_Recall\n";
    
    // Dataset path
    const std::string dataset_path = "/home/wjl/fanns-benchmark/datasets/arxiv-for-fanns-medium";
    
    log_output(logfile, "=== Detailed Profiling: Faiss HNSW vs Navix ===\n");
    log_output(logfile, "Dataset path: " + dataset_path + "\n\n");
    
    // Step 1: Load data
    log_output(logfile, "Step 1: Loading dataset...\n");
    
    auto db_vectors = read_fvecs(dataset_path + "/database_vectors.fvecs");
    if (db_vectors.empty()) return -1;
    
    auto query_vectors = read_fvecs(dataset_path + "/query_vectors.fvecs");
    if (query_vectors.empty()) return -1;
    
    auto db_labels = read_jsonl_labels(dataset_path + "/database_attributes.jsonl", "number_of_sub_categories");
    if (db_labels.empty()) return -1;

    auto query_labels = read_jsonl_labels(dataset_path + "/em_query_attributes.jsonl", "label");
    if (query_labels.empty()) return -1;

    auto ground_truth = read_ivecs(dataset_path + "/ground_truth_em.ivecs");
    if (ground_truth.empty()) return -1;
    
    // Get dataset info
    int dimension = db_vectors[0].size();
    int num_data_points = db_vectors.size();
    int num_queries = std::min(100, (int)query_vectors.size()); // 限制测试数量
    int k = ground_truth[0].size();
    int M = 32;
    
    log_output(logfile, "Dataset info:\n");
    log_output(logfile, "- Database vectors: " + std::to_string(num_data_points) + "\n");
    log_output(logfile, "- Query vectors (testing): " + std::to_string(num_queries) + "\n");
    log_output(logfile, "- Dimension: " + std::to_string(dimension) + "\n");
    log_output(logfile, "- k: " + std::to_string(k) + "\n\n");
    
    // Step 2: Initialize and build indexes
    log_output(logfile, "Step 2: Building indexes...\n");
    
    faiss::IndexHNSWFlat navix_index(dimension, M, faiss::METRIC_L2);
    navix_index.hnsw.efConstruction = 200;
    navix_index.hnsw.efSearch = 200;
    
    faiss::IndexHNSWFlat hnsw_index(dimension, M, faiss::METRIC_L2);
    hnsw_index.hnsw.efConstruction = 200;
    hnsw_index.hnsw.efSearch = 200;
    
    auto db_data_flat = flatten_vectors(db_vectors);
    navix_index.add(num_data_points, db_data_flat.data());
    hnsw_index.add(num_data_points, db_data_flat.data());
    
    auto query_data_flat = flatten_vectors(query_vectors);
    
    log_output(logfile, "Indexes built successfully.\n\n");
    
    // Step 3: 详细性能分析
    log_output(logfile, "Step 3: Detailed Performance Analysis...\n");
    log_output(logfile, "==================================================================================\n");
    log_output(logfile, "Query | Label | Sel. | Navix μs(Prep/Search/Filter/Post) | HNSW μs(Prep/Search/Filter/Post) | Stats\n");
    log_output(logfile, "==================================================================================\n");
    
    // 累计统计
    struct {
        std::chrono::microseconds navix_total, navix_prep, navix_search, navix_filter, navix_post;
        std::chrono::microseconds hnsw_total, hnsw_prep, hnsw_search, hnsw_filter, hnsw_post;
        double navix_recall_sum, hnsw_recall_sum;
        int count;
    } summary = {};
    
    for (int q = 0; q < num_queries; q++) {
        int query_label = query_labels[q];
        
        // 计算选择性
        int filtered_count = 0;
        for (int label : db_labels) {
            if (label == query_label) filtered_count++;
        }
        if (filtered_count == 0) continue;
        
        double selectivity = (double)filtered_count / db_labels.size();
        
        std::vector<faiss::idx_t> navix_labels(k), hnsw_labels(k);
        std::vector<float> navix_distances(k), hnsw_distances(k);
        
        // Navix 性能分析
        auto navix_timing = profile_navix_search(
            navix_index, query_data_flat.data() + q * dimension, k, dimension,
            db_labels, query_label, navix_labels, navix_distances);
        
        // HNSW+Filter 性能分析
        auto hnsw_timing = profile_hnsw_filter_search(
            hnsw_index, query_data_flat.data() + q * dimension, k, dimension,
            db_labels, query_label, hnsw_labels, hnsw_distances);
        
        // 计算召回率
        float navix_recall = calculate_recall(ground_truth[q], navix_labels);
        float hnsw_recall = calculate_recall(ground_truth[q], hnsw_labels);
        
        // 输出到日志文件
        std::stringstream line;
        line << std::setw(5) << q << " | " 
             << std::setw(5) << query_label << " | "
             << std::setw(4) << std::fixed << std::setprecision(2) << selectivity << " | "
             << navix_timing.total_time.count() << "(" 
             << navix_timing.preparation_time.count() << "/"
             << navix_timing.search_core_time.count() << "/"
             << navix_timing.filtering_time.count() << "/"
             << navix_timing.postprocess_time.count() << ") | "
             << hnsw_timing.total_time.count() << "(" 
             << hnsw_timing.preparation_time.count() << "/"
             << hnsw_timing.search_core_time.count() << "/"
             << hnsw_timing.filtering_time.count() << "/"
             << hnsw_timing.postprocess_time.count() << ") | "
             << "R:" << std::setprecision(3) << navix_recall << "/" << hnsw_recall << "\n";
        log_output(logfile, line.str());
        
        // 写入CSV
        csvfile << q << "," << query_label << "," << selectivity << ","
                << navix_timing.total_time.count() << "," << navix_timing.preparation_time.count() << ","
                << navix_timing.search_core_time.count() << "," << navix_timing.filtering_time.count() << ","
                << navix_timing.postprocess_time.count() << ","
                << hnsw_timing.total_time.count() << "," << hnsw_timing.preparation_time.count() << ","
                << hnsw_timing.search_core_time.count() << "," << hnsw_timing.filtering_time.count() << ","
                << hnsw_timing.postprocess_time.count() << ","
                << navix_timing.nodes_visited << "," << navix_timing.distance_computations << ","
                << navix_timing.candidates_evaluated << "," << navix_timing.filtered_out << ","
                << hnsw_timing.candidates_evaluated << "," << hnsw_timing.filtered_out << ","
                << navix_recall << "," << hnsw_recall << "\n";
        
        // 累计统计
        summary.navix_total += navix_timing.total_time;
        summary.navix_prep += navix_timing.preparation_time;
        summary.navix_search += navix_timing.search_core_time;
        summary.navix_filter += navix_timing.filtering_time;
        summary.navix_post += navix_timing.postprocess_time;
        
        summary.hnsw_total += hnsw_timing.total_time;
        summary.hnsw_prep += hnsw_timing.preparation_time;
        summary.hnsw_search += hnsw_timing.search_core_time;
        summary.hnsw_filter += hnsw_timing.filtering_time;
        summary.hnsw_post += hnsw_timing.postprocess_time;
        
        summary.navix_recall_sum += navix_recall;
        summary.hnsw_recall_sum += hnsw_recall;
        summary.count++;
    }
    
    log_output(logfile, "==================================================================================\n");
    
    // 输出汇总分析
    if (summary.count > 0) {
        log_output(logfile, "\n=== DETAILED PERFORMANCE BREAKDOWN ===\n");
        log_output(logfile, "Queries analyzed: " + std::to_string(summary.count) + "\n\n");
        
        log_output(logfile, "NAVIX Average Time Breakdown (μs):\n");
        log_output(logfile, "- Preparation: " + std::to_string(summary.navix_prep.count() / summary.count) + "\n");
        log_output(logfile, "- Core Search: " + std::to_string(summary.navix_search.count() / summary.count) + "\n");
        log_output(logfile, "- Filtering: " + std::to_string(summary.navix_filter.count() / summary.count) + " (integrated)\n");
        log_output(logfile, "- Post-process: " + std::to_string(summary.navix_post.count() / summary.count) + "\n");
        log_output(logfile, "- TOTAL: " + std::to_string(summary.navix_total.count() / summary.count) + "\n\n");
        
        log_output(logfile, "HNSW+Filter Average Time Breakdown (μs):\n");
        log_output(logfile, "- Preparation: " + std::to_string(summary.hnsw_prep.count() / summary.count) + "\n");
        log_output(logfile, "- Core Search: " + std::to_string(summary.hnsw_search.count() / summary.count) + "\n");
        log_output(logfile, "- Filtering: " + std::to_string(summary.hnsw_filter.count() / summary.count) + " (callback overhead)\n");
        log_output(logfile, "- Post-process: " + std::to_string(summary.hnsw_post.count() / summary.count) + "\n");
        log_output(logfile, "- TOTAL: " + std::to_string(summary.hnsw_total.count() / summary.count) + "\n\n");
        
        double speedup = (double)summary.hnsw_total.count() / summary.navix_total.count();
        log_output(logfile, "Overall Speedup: " + std::to_string(speedup) + "x\n");
        log_output(logfile, "Average Recall - Navix: " + std::to_string(summary.navix_recall_sum / summary.count) + 
                  ", HNSW: " + std::to_string(summary.hnsw_recall_sum / summary.count) + "\n");
    }
    
    logfile.close();
    csvfile.close();
    
    std::cout << "\nDetailed profiling logs saved to: " << log_ss.str() << std::endl;
    std::cout << "CSV data saved to: " << csv_ss.str() << std::endl;
    
    return 0;
}