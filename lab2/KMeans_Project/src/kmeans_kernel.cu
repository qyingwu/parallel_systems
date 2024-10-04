#include "kmeans.h"
#include "atomic_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

// CUDA Kernels Implementation

__global__ void assign_points_to_centroids(const double* d_points, const double* d_centroids, int* d_labels, int num_points, int k, int dims) {
    extern __shared__ double s_centroids[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    for (int c = tid; c < k * dims; c += blockDim.x) {
        s_centroids[c] = d_centroids[c];
    }
    __syncthreads();

    if (idx < num_points) {
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
}

__global__ void compute_new_centroids(const double* d_points, const int* d_labels, double* d_centroids, int* d_cluster_sizes, int num_points, int k, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        int cluster_id = d_labels[idx];
        for (int d = 0; d < dims; ++d) {
            atomicAdd(&d_centroids[cluster_id * dims + d], d_points[idx * dims + d]);
        }
        atomicAdd(&d_cluster_sizes[cluster_id], 1);
    }
}

__global__ void normalize_centroids(double* d_centroids, const int* d_cluster_sizes, int k, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < k) {
        for (int d = 0; d < dims; ++d) {
            if (d_cluster_sizes[idx] > 0) {
                d_centroids[idx * dims + d] /= d_cluster_sizes[idx];
            }
        }
    }
}

__global__ void compute_change(double* d_centroids, double* d_old_centroids, double* d_change, int k, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k * dims) {
        double diff = d_centroids[idx] - d_old_centroids[idx];
        d_change[idx] = diff * diff;
    }
}

// Wrapper function for basic CUDA KMeans implementation
void kmeans_cuda(int k, int dims, int max_iters, double threshold, const std::vector<std::vector<double>>& data,
                 std::vector<int>& labels, std::vector<std::vector<double>>& centroids) {

    int num_points = data.size();
    double* d_points;
    double* d_centroids;
    int* d_labels;
    int* d_cluster_sizes;
    double* d_change;

    // Allocate memory on device
    cudaMalloc(&d_points, num_points * dims * sizeof(double));
    cudaMalloc(&d_centroids, k * dims * sizeof(double));
    cudaMalloc(&d_labels, num_points * sizeof(int));
    cudaMalloc(&d_cluster_sizes, k * sizeof(int));
    cudaMalloc(&d_change, k * dims * sizeof(double));

    // Initialize data on host
    std::vector<double> h_points(num_points * dims);
    std::vector<double> h_centroids(k * dims);
    std::vector<int> h_labels(num_points);
    std::vector<int> h_cluster_sizes(k, 0);

    for (int i = 0; i < num_points; ++i) {
        for (int d = 0; d < dims; ++d) {
            h_points[i * dims + d] = data[i][d];
        }
    }

    initialize_centroids(k, data, centroids, time(0));
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < dims; ++d) {
            h_centroids[i * dims + d] = centroids[i][d];
        }
    }

    // Copy data to device
    cudaMemcpy(d_points, h_points.data(), num_points * dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids.data(), k * dims * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_cluster_sizes, 0, k * sizeof(int));

    double* d_old_centroids;
    cudaMalloc(&d_old_centroids, k * dims * sizeof(double));
    cudaMemcpy(d_old_centroids, d_centroids, k * dims * sizeof(double), cudaMemcpyDeviceToDevice);

    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;
    int sharedMemSize = k * dims * sizeof(double);

    for (int iter = 0; iter < max_iters; ++iter) {
        std::cout << "Running iteration " << iter + 1 << std::endl;

        assign_points_to_centroids<<<numBlocks, blockSize, sharedMemSize>>>(d_points, d_centroids, d_labels, num_points, k, dims);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        cudaMemset(d_cluster_sizes, 0, k * sizeof(int));
        compute_new_centroids<<<numBlocks, blockSize>>>(d_points, d_labels, d_centroids, d_cluster_sizes, num_points, k, dims);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        normalize_centroids<<<(k + blockSize - 1) / blockSize, blockSize>>>(d_centroids, d_cluster_sizes, k, dims);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Convergence check
        cudaMemset(d_change, 0, k * dims * sizeof(double));
        compute_change<<<(k * dims + blockSize - 1) / blockSize, blockSize>>>(d_centroids, d_old_centroids, d_change, k, dims);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        std::vector<double> h_change(k * dims);
        cudaMemcpy(h_change.data(), d_change, k * dims * sizeof(double), cudaMemcpyDeviceToHost);

        double change = 0.0;
        for (double ch : h_change) {
            change += ch;
        }

        if (sqrt(change) < threshold) {
            std::cout << "Converged at iteration " << iter + 1 << std::endl;
            break;
        }

        cudaMemcpy(d_old_centroids, d_centroids, k * dims * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(h_centroids.data(), d_centroids, k * dims * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_labels.data(), d_labels, num_points * sizeof(int), cudaMemcpyDeviceToHost);

    // Convert host centroids back to 2D vector
    for (int i = 0; i < k; ++i) {
        centroids[i].resize(dims);
        for (int d = 0; d < dims; ++d) {
            centroids[i][d] = h_centroids[i * dims + d];
        }
    }

    labels.assign(h_labels.begin(), h_labels.end());

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_cluster_sizes);
    cudaFree(d_change);
    cudaFree(d_old_centroids);
}
