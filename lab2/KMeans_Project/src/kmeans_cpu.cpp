#include "kmeans.h"
#include "distance.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <ctime>
#include <cmath> 

// Helper function to find the closest centroid
int find_closest_centroid(const std::vector<double>& point, const std::vector<std::vector<double>>& centroids) {
    double min_dist = std::numeric_limits<double>::max();
    int closest_centroid = -1;

    for (int c = 0; c < centroids.size(); ++c) {
        double dist = euclidean_distance(point, centroids[c]);
        if (dist < min_dist) {
            min_dist = dist;
            closest_centroid = c;
        }
    }
    return closest_centroid;
}

// Sequential CPU-based KMeans implementation
void kmeans_cpu(int k, int dims, int max_iters, double threshold, const std::vector<std::vector<double>> &data, 
                std::vector<int> &labels, std::vector<std::vector<double>> &centroids, int& actual_iters) {
    int n_points = data.size();
    
    std::vector<std::vector<double>> new_centroids(k, std::vector<double>(dims, 0.0f));
    std::vector<int> cluster_sizes(k, 0);
    double total_shift = 0.0f;
    actual_iters = 0;  // Initialize actual iterations
    for (int iter = 0; iter < max_iters; ++iter) {
        actual_iters = iter+1;
        total_shift = 0.0f;

        // Step 1: Assign each point to the nearest centroid
        for (int i = 0; i < n_points; ++i) {
            labels[i] = find_closest_centroid(data[i], centroids);
        }

        // Step 2: Recalculate the centroids based on the assignments
        new_centroids.assign(k, std::vector<double>(dims, 0.0f));
        cluster_sizes.assign(k, 0);

        for (int i = 0; i < n_points; ++i) {
            int cluster_id = labels[i];
            for (int d = 0; d < dims; ++d) {
                new_centroids[cluster_id][d] += data[i][d];
            }
            cluster_sizes[cluster_id]++;
        }

        // Normalize the new centroids
        for (int c = 0; c < k; ++c) {
            if (cluster_sizes[c] > 0) {
                for (int d = 0; d < dims; ++d) {
                    new_centroids[c][d] /= cluster_sizes[c];
                }
            }
        }

        // Step 3: Check for convergence on a per-dimension basis
        bool converged = true; 

        for (int c = 0; c < k; ++c) {
            for (int d = 0; d < dims; ++d) {
                // Check the shift for each dimension
                double dimension_shift = std::abs(centroids[c][d] - new_centroids[c][d]);
                // Print the threshold and the difference (dimension_shift)
                if (dimension_shift > threshold) {
                    converged = false; // If any dimension exceeds the threshold, mark as not converged
                }
            }
        }

        centroids = new_centroids;
        
        // Stop if all dimensions for all centroids have converged
        if (converged) {
            //std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }
}