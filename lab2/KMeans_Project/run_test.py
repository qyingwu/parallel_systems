import numpy as np
from scipy.spatial.distance import cdist
import os


def remap_centroids(iCenters, refCenters, strategy="Min", threshold=1e-4):
    k = iCenters.shape[0]
    distances_matrix = cdist(iCenters, refCenters, metric='euclidean')
    
    matched_indices = np.argmin(distances_matrix, axis=1)
    used_ref_centroids = set()

    for i in range(k):
        if matched_indices[i] in used_ref_centroids:
            if strategy == "Cur":
                current_iC = i
            elif strategy == "Min":
                current_iC = np.argmin(distances_matrix[:, matched_indices[i]])
            elif strategy == "Max":
                current_iC = np.argmax(distances_matrix[:, matched_indices[i]])

            remaining_ref_indices = [j for j in range(k) if j not in used_ref_centroids]
            new_ref_index = remaining_ref_indices[np.argmin(distances_matrix[current_iC, remaining_ref_indices])]
            matched_indices[current_iC] = new_ref_index
        
        used_ref_centroids.add(matched_indices[i])
    
    incorrect_count = 0
    final_distances = []
    for i in range(k):
        dist = distances_matrix[i, matched_indices[i]]
        final_distances.append(dist)
        if dist > threshold:
            incorrect_count += 1

    return matched_indices, incorrect_count, final_distances


def test_kmeans_output(iCenters, refCenters, strategies=["Cur", "Min", "Max"], threshold=1e-4):
    incorrect_counts = {}
    best_strategy = None
    fewest_incorrect = len(iCenters) + 1
    best_distances = []

    for strategy in strategies:
        _, incorrect_count, distances = remap_centroids(iCenters, refCenters, strategy=strategy, threshold=threshold)
        incorrect_counts[strategy] = incorrect_count
        if incorrect_count < fewest_incorrect:
            fewest_incorrect = incorrect_count
            best_strategy = strategy
            best_distances = distances

    return best_strategy, incorrect_counts, best_distances


def load_centroids(filename):
    """
    Load centroids from a text file where each line represents one centroid.
    """
    centroids = []
    with open(filename, 'r') as file:
        for line in file:
            centroids.append([float(x) for x in line.strip().split()])
    return np.array(centroids)


def run_tests(input_folder, tests_folder, threshold=1e-4):
    """
    Run KMeans tests by reading input files, running KMeans, and comparing with reference centroids.
    """
    input_files = [f for f in os.listdir(input_folder) if f.endswith('_input.txt')]
    
    for input_file in input_files:
        test_name = input_file.replace('_input.txt', '')
        input_path = os.path.join(input_folder, input_file)
        ref_centroid_path = os.path.join(tests_folder, f"{test_name}_ref_centroids.txt")

        print(f"\nRunning test: {test_name}")

        # Load data points from input file
        data = load_centroids(input_path)  # Assuming the input data is in the same format as the centroids

        # Load reference centroids from test folder
        ref_centroids = load_centroids(ref_centroid_path)

        # Here, you would run your KMeans algorithm using your `kmeans_cpu` function
        # Assuming your KMeans CPU function returns the centroids calculated
        k = ref_centroids.shape[0]
        dims = ref_centroids.shape[1]
        labels = np.zeros(len(data))  # Placeholder, as labels are irrelevant for this test
        centroids = []  # Centroids will be populated by your kmeans_cpu function
        iterations_run = 0
        
        # Run your KMeans CPU function
        kmeans_cpu(k, dims, 100, threshold, data, labels, centroids, iterations_run)

        # Convert centroids to np array for comparison
        iCenters = np.array(centroids)

        # Compare the centroids from KMeans with reference centroids
        best_strategy, incorrect_counts, best_distances = test_kmeans_output(iCenters, ref_centroids, threshold=threshold)

        # Output the result of the test
        print(f"Best remapping strategy: {best_strategy}")
        print(f"Incorrect centroids count by strategy: {incorrect_counts}")
        print(f"Distances with best strategy: {best_distances}")


# Example usage
if __name__ == "__main__":
    input_folder = "input/"  # Path to your input folder
    tests_folder = "tests/"  # Path to your tests folder
    run_tests(input_folder, tests_folder)
