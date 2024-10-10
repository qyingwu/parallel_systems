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
CUDA_OBJS = src/kmeans_gmem.o src/kmeans_shmem.o src/kmeans_thrust.o

# Target for building the executable
all: bin/kmeans

# Rule to link the final executable with nvcc
bin/kmeans: $(OBJS) $(CUDA_OBJS)
	$(CXX) $(CXXFLAGS) -o bin/kmeans $(OBJS) $(CUDA_OBJS) $(LIBS)

src/kmeans_cpu.o: src/kmeans_cpu.cpp
	$(CXX) $(CXXFLAGS) -c src/kmeans_cpu.cpp -o src/kmeans_cpu.o

src/kmeans.o: src/kmeans.cpp
	$(NVCC) $(NVCCFLAGS) -c src/kmeans.cpp -o src/kmeans.o $(INCLUDES)

src/kmeans_gmem.o: src/kmeans_gmem.cu
	$(NVCC) $(NVCCFLAGS) -c src/kmeans_gmem.cu -o src/kmeans_gmem.o $(INCLUDES)

src/kmeans_shmem.o: src/kmeans_shmem.cu
	$(NVCC) $(NVCCFLAGS) -c src/kmeans_shmem.cu -o src/kmeans_shmem.o $(INCLUDES)

src/kmeans_thrust.o: src/kmeans_thrust.cu
	$(NVCC) $(NVCCFLAGS) -c src/kmeans_thrust.cu -o src/kmeans_thrust.o $(INCLUDES)

# Clean the build
clean:
	rm -f src/*.o bin/kmeans
