/*
#include "kmeans.h"
#include "atomic_utils.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

// CUDA kernel to compute the distances between each point and the centroids
__global__ void compute_distances(const double* d_points, const double* d_centroids, int* d_labels, int num_points, int k, int dims) {
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

// CUDA kernel to compute new centroids
__global__ void compute_new_centroids(const double* d_points, double* d_centroids, int* d_labels, int* d_cluster_sizes, int num_points, int k, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points) return;

    int cluster_id = d_labels[idx];
    for (int d = 0; d < dims; ++d) {
        atomicAdd(&d_centroids[cluster_id * dims + d], d_points[idx * dims + d]);
    }
    atomicAdd(&d_cluster_sizes[cluster_id], 1);
}

// Main function to perform KMeans on GPU using CUDA
void kmeans_cuda(int k, int dims, int max_iters, double threshold, const std::vector<std::vector<double>> &data, 
                 std::vector<int> &labels, std::vector<std::vector<double>> &centroids) {

    int num_points = data.size();

    // Allocate host memory
    double* h_points = new double[num_points * dims];
    double* h_centroids = new double[k * dims];
    int* h_labels = new int[num_points];
    int* h_cluster_sizes = new int[k];

    // Flatten the 2D data vector into a 1D array for CUDA
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
    double* d_points;
    double* d_centroids;
    int* d_labels;
    int* d_cluster_sizes;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_points, num_points * dims * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, k * dims * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_labels, num_points * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_cluster_sizes, k * sizeof(int)));

    // Copy points to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_points, h_points, num_points * dims * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;
    int sharedMemSize = k * dims * sizeof(double);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int iter = 0; iter < max_iters; ++iter) {
        cudaEventRecord(start);

        // Debugging statement to print current iteration
        std::cout << "Running iteration " << iter + 1 << " of " << max_iters << std::endl;

        // Copy centroids to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, k * dims * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemset(d_cluster_sizes, 0, k * sizeof(int)));

        // Step 1: Assign points to closest centroids (now using shared memory)
        compute_distances<<<numBlocks, blockSize, sharedMemSize>>>(d_points, d_centroids, d_labels, num_points, k, dims);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Step 2: Recompute centroids by averaging the points in each cluster
        CHECK_CUDA_ERROR(cudaMemset(d_centroids, 0, k * dims * sizeof(double)));
        compute_new_centroids<<<numBlocks, blockSize>>>(d_points, d_centroids, d_labels, d_cluster_sizes, num_points, k, dims);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Copy centroids and cluster sizes back to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, k * dims * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_cluster_sizes, d_cluster_sizes, k * sizeof(int), cudaMemcpyDeviceToHost));

        // Normalize centroids by the number of points assigned to each cluster
        for (int c = 0; c < k; ++c) {
            if (h_cluster_sizes[c] > 0) {
                for (int d = 0; d < dims; ++d) {
                    h_centroids[c * dims + d] /= h_cluster_sizes[c];
                }
            }
            // Handle the case where no points are assigned to the cluster
            else {
                std::cout << "Cluster " << c << " is empty." << std::endl;
            }
        }

        cudaEventRecord(stop);
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        std::cout << "Iteration " << iter + 1 << " completed in " << milliseconds << " ms" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy labels back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_labels, d_labels, num_points * sizeof(int), cudaMemcpyDeviceToHost));

    // Copy centroids back to output vector
    centroids.clear();
    for (int c = 0; c < k; ++c) {
        std::vector<double> centroid(dims);
        for (int d = 0; d < dims; ++d) {
            centroid[d] = h_centroids[c * dims + d];
        }
        centroids.push_back(centroid);
    }

    // Print the final centroids
    std::cout << "Final centroids:" << std::endl;
    for (int c = 0; c < k; ++c) {
        std::cout << "Centroid " << c << ": ";
        for (int d = 0; d < dims; ++d) {
            std::cout << centroids[c][d] << " ";
        }
        std::cout << std::endl;
    }

    labels.clear();
    for (int i = 0; i < num_points; ++i) {
        labels.push_back(h_labels[i]);
    }

    // Free memory
    delete[] h_points;
    delete[] h_centroids;
    delete[] h_labels;
    delete[] h_cluster_sizes;
    CHECK_CUDA_ERROR(cudaFree(d_points));
    CHECK_CUDA_ERROR(cudaFree(d_centroids));
    CHECK_CUDA_ERROR(cudaFree(d_labels));
    CHECK_CUDA_ERROR(cudaFree(d_cluster_sizes));
}
*/