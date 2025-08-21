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
    int k = 10;
    
    std::cout << "Testing single query Navix search" << std::endl;
    
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
    index.add(num_data_points, data.data());
    
    // 生成查询点
    std::vector<float> query(dimension);
    for (int i = 0; i < dimension; i++) {
        query[i] = dis(gen);
    }
    
    // 生成过滤掩码（50%选择性）
    std::vector<uint8_t> filter_mask(num_data_points);
    std::uniform_real_distribution<float> uniform(0.0, 1.0);
    for (int i = 0; i < num_data_points; i++) {
        filter_mask[i] = (uniform(gen) < 0.5) ? 1 : 0;
    }
    
    // 分配结果数组
    std::vector<faiss::idx_t> navix_labels(k);
    std::vector<float> navix_distances(k);
    std::vector<faiss::idx_t> regular_labels(k);
    std::vector<float> regular_distances(k);
    
    // 创建访问表和统计对象
    faiss::VisitedTable visited_navix(num_data_points);
    faiss::VisitedTable visited_regular(num_data_points);
    faiss::HNSWStats stats_navix;
    faiss::HNSWStats stats_regular;
    
    // 测试 Navix 单查询搜索
    auto start = std::chrono::high_resolution_clock::now();
    index.navix_single_search(
        query.data(), 
        k, 
        navix_distances.data(), 
        navix_labels.data(), 
        reinterpret_cast<char*>(filter_mask.data()),
        visited_navix,
        stats_navix
    );
    auto end = std::chrono::high_resolution_clock::now();
    auto navix_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试常规单查询搜索
    start = std::chrono::high_resolution_clock::now();
    index.single_search(
        query.data(),
        k,
        regular_distances.data(),
        regular_labels.data(),
        visited_regular,
        stats_regular
    );
    end = std::chrono::high_resolution_clock::now();
    auto regular_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Results:" << std::endl;
    std::cout << "  Navix search time: " << navix_time.count() << " μs" << std::endl;
    std::cout << "  Regular search time: " << regular_time.count() << " μs" << std::endl;
    std::cout << "  Navix distance computations: " << stats_navix.ndis << std::endl;
    std::cout << "  Regular distance computations: " << stats_regular.ndis << std::endl;
    std::cout << "  Navix hops: " << stats_navix.nhops << std::endl;
    std::cout << "  Regular hops: " << stats_regular.nhops << std::endl;
    
    // 显示前几个结果
    std::cout << "\nNavix results (first 5):" << std::endl;
    for (int i = 0; i < std::min(5, k); i++) {
        std::cout << "  ID: " << navix_labels[i] << ", Distance: " << navix_distances[i] << std::endl;
    }
    
    return 0;
}