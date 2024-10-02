#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <string>


void initialize_centroids(int k, const std::vector<std::vector<double>>& data, 
                          std::vector<std::vector<double>>& centroids, int seed);

// Function to perform KMeans clustering sequentially on the CPU
void kmeans_cpu(int k, int dims, int max_iters, double threshold, 
                const std::vector<std::vector<double>>& data, 
                std::vector<int>& labels, 
                std::vector<std::vector<double>>& centroids, 
                int& iterations_run);

// Function to perform KMeans clustering using CUDA
void kmeans_cuda(int k, int dims, int max_iters, double threshold, 
                 const std::vector<std::vector<double>>& data, 
                 std::vector<int>& labels, 
                 std::vector<std::vector<double>>& centroids);

// Function to perform KMeans clustering using Thrust
void kmeans_thrust(int k, int dims, int max_iters, double threshold, 
                   const std::vector<std::vector<double>>& data, 
                   std::vector<int>& labels, 
                   std::vector<std::vector<double>>& centroids);

// Function to read input file into a vector of points
std::vector<std::vector<double>> read_input_file(const std::string& filename, int dims);

// Wrapper function to run KMeans based on user selection (CPU, CUDA, or Thrust)
void run_kmeans(int k, int dims, int max_iters, double threshold, 
                bool output_centroids, int seed, 
                bool use_cuda, bool use_thrust, 
                const std::string& input_file);

#endif // KMEANS_H
