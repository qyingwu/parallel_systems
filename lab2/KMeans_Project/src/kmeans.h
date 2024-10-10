
#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <string>

// Function to initialize centroids randomly using a specified seed
void initialize_centroids(int k, const std::vector<std::vector<double>>& data,
                          std::vector<std::vector<double>>& centroids, int seed);

// Function to perform KMeans clustering sequentially on the CPU
void kmeans_cpu(int k, int dims, int max_iters, double threshold,
                const std::vector<std::vector<double>>& data,
                std::vector<int>& labels,
                std::vector<std::vector<double>>& centroids);

// Function to perform KMeans clustering using CUDA gmem
void kmeans_cuda_gmem(int k, int dims, int max_iters, double threshold,
                 const std::vector<std::vector<double>>& data,
                 std::vector<int>& labels,
                 std::vector<std::vector<double>>& centroids);

// Function to perform KMeans clustering using CUDA shmem
void kmeans_cuda_shmem(int k, int dims, int max_iters, double threshold,
                 const std::vector<std::vector<double>>& data,
                 std::vector<int>& labels,
                 std::vector<std::vector<double>>& centroids);
                 

// Function to perform KMeans clustering using Thrust
void kmeans_thrust(int k, int dims, int max_iters, double threshold,
                   const std::vector<std::vector<double>>& data, 
                   std::vector<int>& labels, std::vector<std::vector<double>>& centroids);

// Function to read input file into a vector of points
std::vector<std::vector<double>> read_input_file(const std::string& filename, int dims);

#endif // KMEANS_H
