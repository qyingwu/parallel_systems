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
#include <iomanip>
#include <unordered_set>

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

int generate_random_index(int n_points) {
    int limit = (kmeans_rmax + 1) - ((kmeans_rmax + 1) % n_points);
    int random_value;
    
    do {
        random_value = kmeans_rand();
    } while (random_value >= limit);  // Reject values that would introduce bias

    return random_value % n_points;
}


// Function to read input file into a vector of points
std::vector<std::vector<double>> read_input_file(const std::string &filename, int dims) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    //data vector of points to be used for kmeans
    std::vector<std::vector<double>> data;
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double> point(dims);
        
        // Skip the ID (first element)
        double id;
        iss >> id;  // Skip this value, assuming it's an ID
        
        // Parse the feature values
        for (int d = 0; d < dims; ++d) {
            iss >> point[d];
        }
        data.push_back(point);
    }
    infile.close();
    return data;
}

// Function to initialize centroids with seed provided by cmd line
//randomly generate k integer numbers between 0 and num_points and 
// use these k integers as indices of points used as initial centroids.
void initialize_centroids(int k, const std::vector<std::vector<double>>& data, 
                          std::vector<std::vector<double>>& centroids, int seed) {
    int n_points = data.size();
    kmeans_srand(seed); // Seed the random number generator

    if (k > n_points) {
        std::cerr << "Error: The number of centroids (k) cannot be greater than the number of data points." << std::endl;
        exit(EXIT_FAILURE);  // or handle by setting k = n_points;
    }

    // Preallocate space for centroids
    centroids.reserve(k);

    // Use unordered_set for O(1) duplicate checking
    std::unordered_set<int> chosen_indices; 

    // Select k unique random points as centroids
    while (centroids.size() < k) {
        int index = kmeans_rand() % n_points;
        
        // Only add the index if it hasn't been used already
        if (chosen_indices.find(index) == chosen_indices.end()) {
            centroids.push_back(data[index]);  // Set the point as a centroid
            chosen_indices.insert(index);     // Mark this index as used
        }
    }
}


// Wrapper function to call the appropriate KMeans implementation (CPU, CUDA gemem, CUDA shmem, or Thrust)
void run_kmeans(int k, int dims, int max_iters, double threshold, bool output_centroids, int seed, 
                bool use_cuda_gmem, bool use_cuda_shmem, bool use_thrust, bool use_cpu, const std::string &input_file) {

    // Read input data from file
    std::vector<std::vector<double>> data = read_input_file(input_file, dims);
    int n_points = data.size();

    // Initialize labels and centroids
    std::vector<int> labels(n_points, -1);
    std::vector<std::vector<double>> centroids;

    // Initialize centroids using custom random number generator and seed
    initialize_centroids(k, data, centroids, seed);

    // Start measuring the time
    auto start = std::chrono::high_resolution_clock::now();

    if (use_cuda_gmem) {
        std::cout << "Running KMeans with CUDA gmem..." << std::endl;
        kmeans_cuda_gmem(k, dims, max_iters, threshold, data, labels, centroids);
    } else if (use_cuda_shmem) {
        std::cout << "Running KMeans with CUDA shmem..." << std::endl;
        kmeans_cuda_shmem(k, dims, max_iters, threshold, data, labels, centroids);
    } else if (use_thrust) {
        std::cout << "Running KMeans with Thrust..." << std::endl;
        kmeans_thrust(k, dims, max_iters, threshold, data, labels, centroids); 
    } else if (use_cpu) {
        std::cout << "Running KMeans with CPU..." << std::endl;
        kmeans_cpu(k, dims, max_iters, threshold, data, labels, centroids);
    } else {
        std::cout << "Running KMeans with CPU..." << std::endl;
        kmeans_cpu(k, dims, max_iters, threshold, data, labels, centroids);
    }

    // End measuring the time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate time elapsed in milliseconds
    std::chrono::duration<double, std::milli> elapsed_time = end - start;
    double total_time = elapsed_time.count();
    std::cout << "Total execution time: " << total_time << " ms" << std::endl;

    // Output results
    int precision = 8; //default to 8
    if (output_centroids) {
        std::cout << "Final centroids:" << std::endl;
        std::cout << std::fixed << std::setprecision(precision);  
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
}


// Function to parse command-line arguments and run KMeans
void run_kmeans_from_cli(int argc, char *argv[]) {
    // Command-line argument parsing
    int k = -1, dims = -1, max_iters = 150;
    double threshold = 1e-8f;
    std::string input_file;
    bool output_centroids = false;
    int seed = 8675309;  // Default seed for random initialization
    bool use_cuda_gmem = false;
    bool use_cuda_shmem = false;
    bool use_thrust = false;
    bool use_cpu = false;

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
            seed = std::atoi(argv[++i]);  // Seed passed from the command line
        } else if (std::string(argv[i]) == "--use_cuda_gmem") {
            use_cuda_gmem = true;
        } else if (std::string(argv[i]) == "--use_cuda_shmem") {
            use_cuda_shmem = true;
        }else if (std::string(argv[i]) == "--use_thrust") {
            use_thrust = true;
        } else if (std::string(argv[i]) == "--use_cpu") {
            use_cpu = true;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // Call run_kmeans to run the chosen KMeans implementation
    run_kmeans(k, dims, max_iters, threshold, output_centroids, seed, use_cuda_gmem, use_cuda_shmem, use_thrust, use_cpu, input_file);
}

// Main function is now inside kmeans.cpp
int main(int argc, char *argv[]) {
    run_kmeans_from_cli(argc, argv);
    return 0;
}