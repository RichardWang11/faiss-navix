CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -I/usr/include
FAISS_ROOT = /home/wjl/faiss-navix
INCLUDES = -I$(FAISS_ROOT) -I$(FAISS_ROOT)/build
LIBS = -L$(FAISS_ROOT)/build/faiss -lfaiss -lblas -llapack -lgomp
test_navix: /home/wjl/faiss-navix/test_navix_cmake/test_navix.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o test_navix /home/wjl/faiss-navix/test_navix_cmake/test_navix.cpp $(LIBS)
test_navix_single: /home/wjl/faiss-navix/test_navix_cmake/test_navix_single.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o test_navix_single /home/wjl/faiss-navix/test_navix_cmake/test_navix_single.cpp $(LIBS)
test_real_data: /home/wjl/faiss-navix/test_navix_cmake/test_real_data.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o test_real_data /home/wjl/faiss-navix/test_navix_cmake/test_real_data.cpp $(LIBS)
all: test_navix test_navix_single test_real_data
clean:
	rm -f test_navix test_navix_single test_real_data
.PHONY: all clean
