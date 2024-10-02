#!/usr/bin/env python2.7
import os
from subprocess import check_output
import re
from time import sleep

# Define the number of clusters, dimensions, iterations, etc.
CLUSTERS = [3, 5, 10]
DIMENSIONS = [2, 16, 32]
ITERATIONS = 100
THRESHOLD = 1e-4
SEED = 42

# Input files to test on (update these with your actual input file paths)
INPUTS = ["random-n2048-d16-c16.txt", "random-n16384-d24-c16.txt", "random-n65536-d32-c16.txt"]

# Test both CPU and CUDA versions
VERSIONS = ["cpu", "cuda"]

# CSV to store results
csvs = []

# Loop over input files, number of clusters, and dimensions
for inp in INPUTS:
    for cluster in CLUSTERS:
        for dim in DIMENSIONS:
            csv = ["{}/{}/{}".format(inp, cluster, dim)]

            for version in VERSIONS:
                # Construct command for CPU or CUDA version
                if version == "cpu":
                    cmd = "./bin/kmeans_cpu -k {} -d {} -i input/{} -m {} -t {} -s {}".format(cluster, dim, inp, ITERATIONS, THRESHOLD, SEED)
                else:
                    cmd = "./bin/kmeans_cpu --use_cuda -k {} -d {} -i input/{} -m {} -t {} -s {}".format(cluster, dim, inp, ITERATIONS, THRESHOLD, SEED)

                try:
                    # Run the command and capture output
                    out = check_output(cmd, shell=True).decode("ascii")
                    # Extract the time per iteration (in milliseconds) from the output
                    m = re.search(r"Time per iteration: ([\d\.]+)", out)
                    if m:
                        time = m.group(1)
                    else:
                        time = "N/A"
                except Exception as e:
                    time = "Error"

                # Append time result to CSV entry
                csv.append(time)

            # Store this CSV row
            csvs.append(csv)
            # Pause between tests to avoid overwhelming the system
            sleep(0.5)

# Output results as CSV format
header = ["Input/Cluster/Dim"] + ["{} version".format(v) for v in VERSIONS]

print(", ".join(header))
for csv in csvs:
    print(", ".join(csv))
