#include "kmeans.h"
#include "atomic_utils.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

// ------------------------------------------------------------------
// CUDA Kernels
// ------------------------------------------------------------------

// Kernel for assigning points to centroids
__global__ void assign_points_to_centroids(const double* d_points, const double* d_centroids, int* d_labels, int num_points, int k, int dims) {
    extern __shared__ double s_centroids[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load centroids into shared memory
    for (int c = tid; c < k * dims; c += blockDim.x) {
        s_centroids[c] = d_centroids[c];
    }
    __syncthreads();

    if (idx >= num_points) return;

    double min_dist = INFINITY;
    int best_cluster = -1;

    for (int c = 0; c < k; ++c) {
        double dist = 0.0;
        for (int d = 0; d < dims; ++d) {
            double diff = d_points[idx * dims + d] - s_centroids[c * dims + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }

    d_labels[idx] = best_cluster;
}

// Kernel for computing new centroids
__global__ void compute_new_centroids(const double* d_points, const int* d_labels, double* d_centroids, int* d_cluster_sizes, int num_points, int k, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    int cluster_id = d_labels[idx];
    for (int d = 0; d < dims; ++d) {
        atomicAdd(&d_centroids[cluster_id * dims + d], d_points[idx * dims + d]);
    }
    atomicAdd(&d_cluster_sizes[cluster_id], 1);
}

// Kernel for normalizing centroids
__global__ void normalize_centroids(double* d_centroids, const int* d_cluster_sizes, int k, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;

    for (int d = 0; d < dims; ++d) {
        if (d_cluster_sizes[idx] > 0) {
            d_centroids[idx * dims + d] /= d_cluster_sizes[idx];
        }
    }
}

// ------------------------------------------------------------------
// Wrapper Functions
// ------------------------------------------------------------------

// Wrapper function for basic CUDA KMeans implementation
void kmeans_cuda(int k, int dims, int max_iters, double threshold, const std::vector<std::vector<double>> &data,
                 std::vector<int> &labels, std::vector<std::vector<double>> &centroids) {

    int num_points = data.size();
    // Allocate host memory
    thrust::host_vector<double> h_points(num_points * dims);
    thrust::host_vector<double> h_centroids(k * dims);
    thrust::host_vector<int> h_labels(num_points);
    thrust::host_vector<int> h_cluster_sizes(k);

    // Copy data to host vector
    for (int i = 0; i < num_points; ++i) {
        for (int d = 0; d < dims; ++d) {
            h_points[i * dims + d] = data[i][d];
        }
    }

    // Initialize centroids randomly
    initialize_centroids(k, data, centroids, time(0));
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < dims; ++d) {
            h_centroids[i * dims + d] = centroids[i][d];
        }
    }

    // Allocate device memory
    thrust::device_vector<double> d_points = h_points;
    thrust::device_vector<double> d_centroids = h_centroids;
    thrust::device_vector<int> d_labels(num_points);
    thrust::device_vector<int> d_cluster_sizes(k);

    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;
    int sharedMemSize = k * dims * sizeof(double);

    for (int iter = 0; iter < max_iters; ++iter) {
        std::cout << "Running iteration " << iter + 1 << std::endl;

        // Step 1: Assign points to centroids
        assign_points_to_centroids<<<numBlocks, blockSize, sharedMemSize>>>(
            thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_centroids.data()), 
            thrust::raw_pointer_cast(d_labels.data()), num_points, k, dims);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Step 2: Recompute centroids by averaging the points in each cluster
        thrust::fill(d_centroids.begin(), d_centroids.end(), 0.0);
        thrust::fill(d_cluster_sizes.begin(), d_cluster_sizes.end(), 0);
        compute_new_centroids<<<numBlocks, blockSize>>>(
            thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_labels.data()), 
            thrust::raw_pointer_cast(d_centroids.data()), thrust::raw_pointer_cast(d_cluster_sizes.data()), 
            num_points, k, dims);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        
        // Step 3: Normalize centroids
        normalize_centroids<<<(k + blockSize - 1) / blockSize, blockSize>>>(
            thrust::raw_pointer_cast(d_centroids.data()), thrust::raw_pointer_cast(d_cluster_sizes.data()), k, dims);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    // Copy results back to host
    thrust::copy(d_centroids.begin(), d_centroids.end(), h_centroids.begin());
    thrust::copy(d_labels.begin(), d_labels.end(), h_labels.begin());

    // Convert host centroids back to 2D vector
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < dims; ++d) {
            centroids[i][d] = h_centroids[i * dims + d];
        }
    }

    // Copy labels back to output
    labels.assign(h_labels.begin(), h_labels.end());
}

// Wrapper function for Thrust-based CUDA KMeans implementation
void kmeans_thrust(int k, int dims, int max_iters, double threshold, const std::vector<std::vector<double>> &data, 
                   std::vector<int> &labels, std::vector<std::vector<double>> &centroids) {
    // Use existing Thrust logic from the previous kmeans_thrust.cu file
    kmeans_cuda(k, dims, max_iters, threshold, data, labels, centroids);  // Placeholder, replace with Thrust logic
}
