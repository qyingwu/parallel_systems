#include "kmeans.h"
#include "distance.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cstdlib>


// (kmeans_srand and kmeans_rand) to generate initial random centroids
static unsigned long int next = 1;
static const unsigned long kmeans_rmax = 32767;

void kmeans_srand(unsigned int seed) {
    next = seed;
}

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % (kmeans_rmax + 1);
}

// Function to read input file into a vector of points
std::vector<std::vector<double>> read_input_file(const std::string &filename, int dims) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double> point(dims);
        for (int d = 0; d < dims; ++d) {
            iss >> point[d];
        }
        data.push_back(point);
    }
    infile.close();
    return data;
}

// Function to normalize the dataset (Min-Max normalization)
void min_max_normalize(std::vector<std::vector<double>>& data) {
    int n_points = data.size();
    if (n_points == 0) return;  // Check if data is empty
    int dims = data[0].size();

    std::vector<double> min_vals(dims, std::numeric_limits<double>::max());
    std::vector<double> max_vals(dims, std::numeric_limits<double>::lowest());

    // Find the min and max for each dimension
    for (int i = 0; i < n_points; ++i) {
        for (int d = 0; d < dims; ++d) {
            if (data[i][d] < min_vals[d]) min_vals[d] = data[i][d];
            if (data[i][d] > max_vals[d]) max_vals[d] = data[i][d];
        }
    }

    // Normalize the data using Min-Max normalization
    for (int i = 0; i < n_points; ++i) {
        for (int d = 0; d < dims; ++d) {
            if (max_vals[d] != min_vals[d]) {  // Avoid division by zero
                data[i][d] = (data[i][d] - min_vals[d]) / (max_vals[d] - min_vals[d]);
            } else {
                data[i][d] = 0.0; // If max equals min, set normalized value to 0
            }
        }
    }
}

// Initialize centroids using custom random number generation
void initialize_centroids(int k, const std::vector<std::vector<double>>& data, 
                          std::vector<std::vector<double>>& centroids, int seed) {
    int n_points = data.size();
    
    // Initialize the random seed using the provided seed
    kmeans_srand(seed);
    
    // Select k random points as the initial centroids
    std::vector<int> chosen_indices; // Keep track of chosen indices to avoid duplicates
    for (int i = 0; i < k; ++i) {
        int index;
        do {
            // Use kmeans_rand() to generate a random index
            index = kmeans_rand() % n_points;

            // Add error checking to ensure the index is valid
            if (index < 0 || index >= n_points) {
                std::cerr << "Error: Invalid index generated for initial centroid." << std::endl;
                exit(EXIT_FAILURE);
            }

        } while (std::find(chosen_indices.begin(), chosen_indices.end(), index) != chosen_indices.end());
        
        centroids.push_back(data[index]); // Set the chosen point as a centroid
        chosen_indices.push_back(index);  // Keep track of selected points
    }
}

// Wrapper function to call the appropriate KMeans implementation (CPU, CUDA, or Thrust)
void run_kmeans(int k, int dims, int max_iters, double threshold, bool output_centroids, int seed, 
                bool use_cuda, bool use_thrust, const std::string &input_file) {

    // Read input data from file
    std::vector<std::vector<double>> data = read_input_file(input_file, dims);
    
    // Normalize the data (Min-Max normalization)
    min_max_normalize(data);

    int n_points = data.size();

    // Initialize labels and centroids
    std::vector<int> labels(n_points, -1);
    std::vector<std::vector<double>> centroids;

    // Measure elapsed time for KMeans
    auto start = std::chrono::high_resolution_clock::now();

    int iterations_run = 0;

    if (use_cuda) {
        std::cout << "Running KMeans with CUDA..." << std::endl;
        kmeans_cuda(k, dims, max_iters, threshold, data, labels, centroids);
    } else if (use_thrust) {
        std::cout << "Running KMeans with Thrust..." << std::endl;
        kmeans_thrust(k, dims, max_iters, threshold, data, labels, centroids);
    } else {
        std::cout << "Running KMeans with CPU..." << std::endl;
        kmeans_cpu(k, dims, max_iters, threshold, data, labels, centroids, iterations_run);
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Calculate time elapsed in milliseconds
    std::chrono::duration<double, std::milli> elapsed_time = end - start;
    double total_time = elapsed_time.count();
    double time_per_iter = iterations_run > 0 ? total_time / iterations_run : total_time;

    // Output results
    if (output_centroids) {
        std::cout << "Final centroids:" << std::endl;
        for (int cluster_id = 0; cluster_id < k; ++cluster_id) {
            std::cout << "Centroid " << cluster_id << ": ";
            for (int d = 0; d < dims; ++d) {
                std::cout << centroids[cluster_id][d] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        // Output point labels
        std::cout << "Clusters:";
        for (int i = 0; i < n_points; ++i) {
            std::cout << " " << labels[i];
        }
        std::cout << std::endl;
    }

    // Output the time and iterations
    std::cout << "Iterations ran: " << iterations_run << std::endl;
    std::cout << "Total time: " << total_time << " ms" << std::endl;
    if (iterations_run > 0) {
        std::cout << "Time per iteration: " << time_per_iter << " ms" << std::endl;
    }
}

// Function to parse command-line arguments and run KMeans
void run_kmeans_from_cli(int argc, char *argv[]) {
    // Command-line argument parsing
    int k = -1, dims = -1, max_iters = 150;
    double threshold = 1e-4;
    std::string input_file;
    bool output_centroids = false;
    int seed = 8675309;  // Default seed for random initialization
    bool use_cuda = false;
    bool use_thrust = false;

    // Parsing command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-k") {
            k = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "-d") {
            dims = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "-i") {
            input_file = argv[++i];
        } else if (std::string(argv[i]) == "-m") {
            max_iters = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "-t") {
            threshold = std::atof(argv[++i]);
        } else if (std::string(argv[i]) == "-c") {
            output_centroids = true;
        } else if (std::string(argv[i]) == "-s") {
            seed = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--use_cuda") {
            use_cuda = true;
        } else if (std::string(argv[i]) == "--use_thrust") {
            use_thrust = true;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Validate mandatory arguments
    if (k == -1 || dims == -1 || input_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " -k num_clusters -d dims -i input_file [-m max_iters] [-t threshold] [-c] [-s seed] [--use_cuda] [--use_thrust]" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Start measuring the time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Call run_kmeans to run the chosen KMeans implementation
    run_kmeans(k, dims, max_iters, threshold, output_centroids, seed, use_cuda, use_thrust, input_file);

    // End measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;

    // Output time per iteration and number of iterations
    std::cout << "Total time: " << elapsed_time.count() << " ms" << std::endl;
}

// Main function is now inside kmeans.cpp
int main(int argc, char *argv[]) {
    run_kmeans_from_cli(argc, argv);
    return 0;
}
