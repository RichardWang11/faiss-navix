cmake_minimum_required(VERSION 3.17)

project(test_navix)

set(CMAKE_CXX_STANDARD 17)

# 找到 faiss
find_package(faiss REQUIRED)

# 创建测试可执行文件
add_executable(test_navix ../test_navix.cpp)
add_executable(test_navix_single ../test_navix_single.cpp)

# 链接 faiss 库
target_link_libraries(test_navix faiss)
target_link_libraries(test_navix_single faiss)

# 设置编译选项
target_compile_options(test_navix PRIVATE -O3 -march=native)
target_compile_options(test_navix_single PRIVATE -O3 -march=native)