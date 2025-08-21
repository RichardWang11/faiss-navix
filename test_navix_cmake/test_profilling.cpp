#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/MetaIndexes.h>
#include <faiss/utils/distances.h>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <iomanip> 
#include <sstream>
#include <unordered_set>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

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

// Implements IDSelector to filter by a specific label
class LabelSelector : public faiss::IDSelector {
private:
    const std::vector<int>& db_labels_;
    int target_label_;
    
public:
    LabelSelector(const std::vector<int>& db_labels, int target_label) 
        : db_labels_(db_labels), target_label_(target_label) {}
    
    bool is_member(faiss::idx_t id) const override {
        if (id < 0 || id >= db_labels_.size()) {
            return false;
        }
        return db_labels_[id] == target_label_;
    }
};

// Calculates recall for a single query
float calculate_recall(const std::vector<int>& ground_truth, const std::vector<faiss::idx_t>& results) {
    if (ground_truth.empty()) {
        return 1.0f; // Or 0.0f, depending on definition for empty ground truth
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


int main() {
    // --- File Setup ---
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    std::stringstream log_ss;
    log_ss << "navix_comparison_log_" 
           << std::put_time(&tm, "%Y%m%d_%H%M%S") 
           << ".txt";
    
    std::ofstream logfile(log_ss.str());
    if (!logfile.is_open()) {
        std::cerr << "Error: Cannot create log file " << log_ss.str() << std::endl;
        return -1;
    }
    std::cout << "Log file created: " << log_ss.str() << std::endl;

    std::stringstream csv_ss;
    csv_ss << "navix_comparison_results_"
           << std::put_time(&tm, "%Y%m%d_%H%M%S")
           << ".csv";
    std::ofstream csvfile(csv_ss.str());
     if (!csvfile.is_open()) {
        std::cerr << "Error: Cannot create CSV file " << csv_ss.str() << std::endl;
        logfile << "Error: Cannot create CSV file " << csv_ss.str() << std::endl;
        return -1;
    }
    std::cout << "CSV results file created: " << csv_ss.str() << std::endl;

    // Write the header row to the CSV file
    csvfile << "Query,Label,Selectivity,Navix (us),HNSW+Filter (us),Speedup,Navix Recall,HNSW Recall\n";
    
    // Dataset path
    const std::string dataset_path = "/home/wjl/fanns-benchmark/datasets/arxiv-for-fanns-medium";
    
    log_output(logfile, "=== Faiss HNSW vs Navix EM Filter Comparison ===\n");
    log_output(logfile, "Dataset path: " + dataset_path + "\n");
    log_output(logfile, "Log file: " + log_ss.str() + "\n");
    log_output(logfile, "CSV file: " + csv_ss.str() + "\n\n");
    
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

    // Load ground truth data
    auto ground_truth = read_ivecs(dataset_path + "/ground_truth_em.ivecs");
    if (ground_truth.empty()) return -1;
    
    // Get dataset info
    int dimension = db_vectors[0].size();
    int num_data_points = db_vectors.size();
    int num_queries = (int)query_vectors.size();
    int k = ground_truth[0].size(); // Use k from ground truth file
    int M = 32;
    
    log_output(logfile, "Dataset info:\n");
    log_output(logfile, "- Database vectors: " + std::to_string(num_data_points) + "\n");
    log_output(logfile, "- Query vectors: " + std::to_string(num_queries) + "\n");
    log_output(logfile, "- Dimension: " + std::to_string(dimension) + "\n");
    log_output(logfile, "- k (from ground truth): " + std::to_string(k) + "\n\n");
    
    // Step 2: Initialize indexes
    log_output(logfile, "Step 2: Initializing indexes...\n");
    
    faiss::IndexHNSWFlat navix_index(dimension, M, faiss::METRIC_L2);
    navix_index.hnsw.efConstruction = 200;
    navix_index.hnsw.efSearch = 200;
    
    faiss::IndexHNSWFlat hnsw_index(dimension, M, faiss::METRIC_L2);
    hnsw_index.hnsw.efConstruction = 200;
    hnsw_index.hnsw.efSearch = 200;
    
    log_output(logfile, "Both indexes initialized with M=" + std::to_string(M) + 
               ", efConstruction=200, efSearch=200\n\n");
    
    // Step 3: Build indexes
    log_output(logfile, "Step 3: Building indexes...\n");
    
    auto db_data_flat = flatten_vectors(db_vectors);
    
    auto start = std::chrono::high_resolution_clock::now();
    navix_index.add(num_data_points, db_data_flat.data());
    auto end = std::chrono::high_resolution_clock::now();
    auto navix_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    start = std::chrono::high_resolution_clock::now();
    hnsw_index.add(num_data_points, db_data_flat.data());
    end = std::chrono::high_resolution_clock::now();
    auto hnsw_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    log_output(logfile, "Navix index built in " + std::to_string(navix_build_time.count()) + " ms\n");
    log_output(logfile, "HNSW index built in " + std::to_string(hnsw_build_time.count()) + " ms\n\n");
    
    // Step 4: Prepare query data
    log_output(logfile, "Step 4: Preparing query data...\n");
    
    auto query_data_flat = flatten_vectors(query_vectors);
    
    log_output(logfile, "Query data prepared for " + std::to_string(num_queries) + " queries\n\n");
    
    // Step 5: Performance comparison
    log_output(logfile, "Step 5: Performance comparison...\n");
    log_output(logfile, "======================================================================================\n");
    log_output(logfile, "Query | Label | Selectivity | Navix (μs) | HNSW+Filter (μs) | Speedup | Navix R | HNSW R\n");
    log_output(logfile, "======================================================================================\n");
    
    double total_navix_time = 0;
    double total_hnsw_time = 0;
    double total_navix_recall = 0;
    double total_hnsw_recall = 0;
    int valid_queries = 0;
    
    for (int q = 0; q < num_queries; q++) {
        int query_label = query_labels[q];
        
        int filtered_count = 0;
        for (int label : db_labels) {
            if (label == query_label) {
                filtered_count++;
            }
        }
        
        if (filtered_count == 0) continue;
        
        double selectivity = (double)filtered_count / db_labels.size();
        
        std::vector<faiss::idx_t> navix_labels(k);
        std::vector<float> navix_distances(k);
        std::vector<faiss::idx_t> hnsw_labels(k);
        std::vector<float> hnsw_distances(k);
        
        // 1. Navix filtered search
        std::vector<uint8_t> filter_mask(num_data_points);
        for (size_t i = 0; i < db_labels.size(); ++i) {
            filter_mask[i] = (db_labels[i] == query_label) ? 1 : 0;
        }
        
        faiss::VisitedTable vt_navix(num_data_points);
        faiss::HNSWStats stats_navix;
        
        start = std::chrono::high_resolution_clock::now();
        navix_index.navix_single_search(
            query_data_flat.data() + q * dimension, k,
            navix_distances.data(), navix_labels.data(),
            reinterpret_cast<char*>(filter_mask.data()), vt_navix, stats_navix);
        end = std::chrono::high_resolution_clock::now();
        auto navix_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 2. Standard HNSW filtered search
        LabelSelector selector(db_labels, query_label);
        faiss::SearchParametersHNSW search_params;
        search_params.sel = &selector;
        search_params.efSearch = 200;
        
        start = std::chrono::high_resolution_clock::now();
        hnsw_index.search(1, query_data_flat.data() + q * dimension, k,
            hnsw_distances.data(), hnsw_labels.data(), &search_params);
        end = std::chrono::high_resolution_clock::now();
        auto hnsw_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 3. Calculate Recall
        float navix_recall = calculate_recall(ground_truth[q], navix_labels);
        float hnsw_recall = calculate_recall(ground_truth[q], hnsw_labels);

        double speedup = (navix_time.count() > 0) ? (double)hnsw_time.count() / navix_time.count() : 0.0;
        
        // --- Write to Log File ---
        std::stringstream line;
        line << std::setw(5) << q << " | " 
             << std::setw(5) << query_label << " | "
             << std::setw(11) << std::fixed << std::setprecision(3) << selectivity << " | "
             << std::setw(10) << navix_time.count() << " | "
             << std::setw(16) << hnsw_time.count() << " | "
             << std::setw(7) << std::fixed << std::setprecision(2) << speedup << "x | "
             << std::setw(7) << std::fixed << std::setprecision(3) << navix_recall << " | "
             << std::setw(7) << std::fixed << std::setprecision(3) << hnsw_recall << "\n";
        log_output(logfile, line.str());
        
        // --- Write to CSV File ---
        csvfile << q << ","
                << query_label << ","
                << selectivity << ","
                << navix_time.count() << ","
                << hnsw_time.count() << ","
                << speedup << ","
                << navix_recall << ","
                << hnsw_recall << "\n";

        total_navix_time += navix_time.count();
        total_hnsw_time += hnsw_time.count();
        total_navix_recall += navix_recall;
        total_hnsw_recall += hnsw_recall;
        valid_queries++;
    }
    
    log_output(logfile, "======================================================================================\n");
    
    // Summary
    if (valid_queries > 0) {
        double avg_navix_time = total_navix_time / valid_queries;
        double avg_hnsw_time = total_hnsw_time / valid_queries;
        double avg_navix_recall = total_navix_recall / valid_queries;
        double avg_hnsw_recall = total_hnsw_recall / valid_queries;
        double overall_speedup = (avg_navix_time > 0) ? avg_hnsw_time / avg_navix_time : 0.0;
        
        std::stringstream summary;
        summary << "\nSummary:\n"
                << "- Valid queries tested: " << valid_queries << "\n"
                << "- Average Navix time: " << std::fixed << std::setprecision(1) << avg_navix_time << " μs\n"
                << "- Average HNSW+Filter time: " << std::fixed << std::setprecision(1) << avg_hnsw_time << " μs\n"
                << "- Average Navix Recall@"<< k <<": " << std::fixed << std::setprecision(4) << avg_navix_recall << "\n"
                << "- Average HNSW+Filter Recall@"<< k <<": " << std::fixed << std::setprecision(4) << avg_hnsw_recall << "\n"
                << "- Overall speedup: " << std::setprecision(2) << overall_speedup << "x\n";
        
        log_output(logfile, summary.str());
    }
    
    log_output(logfile, "\n=== Comparison completed ===\n");
    
    auto end_time = std::chrono::system_clock::now();
    auto end_time_t = std::chrono::system_clock::to_time_t(end_time);
    auto end_tm = *std::localtime(&end_time_t);
    
    std::stringstream end_time_str;
    end_time_str << "Test completed at: " << std::put_time(&end_tm, "%Y-%m-%d %H:%M:%S") << "\n";
    log_output(logfile, end_time_str.str());
    
    logfile.close();
    csvfile.close();
    
    std::cout << "\nDetailed logs saved to: " << log_ss.str() << std::endl;
    std::cout << "CSV data saved to: " << csv_ss.str() << std::endl;
    
    return 0;
}
