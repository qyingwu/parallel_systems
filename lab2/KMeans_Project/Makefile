# Compiler and flags
NVCC = /usr/local/cuda-10.1/bin/nvcc   # Path to the CUDA compiler
CXX = g++                              # C++ compiler

CXXFLAGS = -O2 -std=c++11              # General C++ flags
NVCCFLAGS = -O2 -arch=sm_50            # Flags for CUDA compilation, adjust for your GPU

# Include paths and libraries for CUDA
INCLUDES = -I/usr/local/cuda-10.1/include -I./src
LIBS = -L/usr/local/cuda-10.1/lib64 -lcudart

# Object files
OBJS = src/kmeans.o src/kmeans_cpu.o
CUDA_OBJS = src/kmeans_kernel.o

# Target for building the executable
all: bin/kmeans

# Rule to link the final executable with nvcc
bin/kmeans: $(OBJS) $(CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) -o bin/kmeans $(OBJS) $(CUDA_OBJS) $(LIBS)

bin/kmeans_cpu.o: 
	$(NVCC) $(NVCCFLAGS) -o src/kmeans_cpu.cpp src/kmeans_cpu.o

# Rule to compile the main kmeans.cpp file (uses nvcc to compile as it may contain CUDA calls)
src/kmeans.o: src/kmeans.cpp
	$(NVCC) $(CXXFLAGS) -c src/kmeans.cpp -o src/kmeans.o $(INCLUDES)

# Rule to compile the CUDA kernel and wrapper function (kmeans_kernel.cu)
src/kmeans_kernel.o: src/kmeans_kernel.cu
	$(NVCC) $(NVCCFLAGS) -c src/kmeans_kernel.cu -o src/kmeans_kernel.o $(INCLUDES)

# Clean the build
clean:
	rm -f src/*.o bin/kmeans
