CXX = g++
#NVCC = /usr/local/cuda-10.1/bin/nvcc  # Path to the CUDA compiler

CXXFLAGS = -O2 -std=c++11
NVCCFLAGS = -O2 -arch=sm_50  # Or the compute capability of your GPU

#INCLUDES = -I/usr/local/cuda-10.1/include -I./src
#LIBS = -L/usr/local/cuda-10.1/lib64 -lcudart

# Object files
OBJS = src/main.o src/kmeans_cpu.o src/kmeans.o
#CUDA_OBJS = src/kmeans_thrust.o src/kmeans_cuda.o  # Include kmeans_cuda.o here

# Executable
all: bin/kmeans_cpu

# Link the final executable with g++
bin/kmeans_cpu: $(OBJS)
	$(CXX) $(CXXFLAGS) -o bin/kmeans_cpu $(OBJS) $(LIBS)

# Compile C++ files with g++
src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDES)

# Compile CUDA files with nvcc
src/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(INCLUDES)

clean:
	rm -f src/*.o bin/kmeans_cpu
