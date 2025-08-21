#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>

int main() {
    // 设置参数
    int dimension = 128;
    int M = 32;
    int num_data_points = 10000;
    int num_queries = 100;
    int k = 10;
    
    std::cout << "Testing Faiss-Navix with:" << std::endl;
    std::cout << "- Dimension: " << dimension << std::endl;
    std::cout << "- Data points: " << num_data_points << std::endl;
    std::cout << "- Queries: " << num_queries << std::endl;
    std::cout << "- k: " << k << std::endl;
    std::cout << std::endl;
    
    // 初始化索引
    faiss::IndexHNSWFlat index(dimension, M, faiss::METRIC_L2);
    index.hnsw.efConstruction = 200;
    index.hnsw.efSearch = 200;
    
    // 生成随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 1.0);
    
    // 生成数据点
    std::vector<float> data(num_data_points * dimension);
    for (int i = 0; i < num_data_points * dimension; i++) {
        data[i] = dis(gen);
    }
    
    // 构建索引
    std::cout << "Building index..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    index.add(num_data_points, data.data());
    auto end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Index built in " << build_time.count() << " ms" << std::endl;
    std::cout << std::endl;
    
    // 生成查询点
    std::vector<float> queries(num_queries * dimension);
    for (int i = 0; i < num_queries * dimension; i++) {
        queries[i] = dis(gen);
    }
    
    // 生成过滤掩码（模拟不同的选择性）
    std::vector<uint8_t> filter_masks(num_data_points * num_queries);
    std::uniform_real_distribution<float> uniform(0.0, 1.0);
    
    // 测试不同的选择性
    std::vector<float> selectivities = {0.1, 0.2, 0.5, 0.8, 1.0};
    
    for (float selectivity : selectivities) {
        std::cout << "Testing with selectivity: " << selectivity << std::endl;
        
        // 为每个查询生成过滤掩码
        for (int i = 0; i < num_queries; i++) {
            for (int j = 0; j < num_data_points; j++) {
                filter_masks[i * num_data_points + j] = (uniform(gen) < selectivity) ? 1 : 0;
            }
        }
        
        // 分配结果数组
        std::vector<faiss::idx_t> navix_labels(num_queries * k);
        std::vector<float> navix_distances(num_queries * k);
        std::vector<faiss::idx_t> regular_labels(num_queries * k);
        std::vector<float> regular_distances(num_queries * k);
        
        // 测试 Navix 搜索
        start = std::chrono::high_resolution_clock::now();
        index.navix_search(
            num_queries, 
            queries.data(), 
            k, 
            navix_distances.data(), 
            navix_labels.data(), 
            reinterpret_cast<char*>(filter_masks.data())
        );
        end = std::chrono::high_resolution_clock::now();
        auto navix_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 测试常规搜索（用于比较）
        start = std::chrono::high_resolution_clock::now();
        index.search(
            num_queries, 
            queries.data(), 
            k, 
            regular_distances.data(), 
            regular_labels.data()
        );
        end = std::chrono::high_resolution_clock::now();
        auto regular_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Navix search time: " << navix_time.count() << " μs" << std::endl;
        std::cout << "  Regular search time: " << regular_time.count() << " μs" << std::endl;
        std::cout << "  Navix QPS: " << (num_queries * 1000000.0) / navix_time.count() << std::endl;
        std::cout << "  Regular QPS: " << (num_queries * 1000000.0) / regular_time.count() << std::endl;
        std::cout << std::endl;
    }
    
    return 0;
}