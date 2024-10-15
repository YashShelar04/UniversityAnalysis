import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from collections import defaultdict

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Single linkage clustering algorithm
def single_linkage_clustering(data):
    clusters = {i: [i] for i in range(len(data))}
    distances = defaultdict(dict)
    
    # Calculate initial pairwise distances
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distances[i][j] = euclidean_distance(data[i], data[j])
    
    dendrogram_steps = []
    cluster_count = len(data)  # Keep track of the new cluster indices
    
    while len(clusters) > 1:
        min_dist = float('inf')
        to_merge = (None, None)
        
        for i in clusters:
            for j in clusters:
                if i != j:
                    dist = min(euclidean_distance(data[p1], data[p2]) 
                               for p1 in clusters[i] for p2 in clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        to_merge = (i, j)
        
        # Merge the two closest clusters
        cluster1, cluster2 = to_merge
        new_cluster = clusters[cluster1] + clusters[cluster2]
        
        # Record the linkage step in dendrogram format
        dendrogram_steps.append([cluster1, cluster2, min_dist, len(new_cluster)])
        
        # Remove the old clusters and assign a new cluster index
        clusters[cluster_count] = new_cluster
        del clusters[cluster1]
        del clusters[cluster2]
        
        cluster_count += 1  # Increment the new cluster index
    
    return clusters, dendrogram_steps

def plot_dendrogram(steps):
    plt.figure(figsize=(8, 6))
    dendrogram(np.array(steps))
    plt.title('Dendrogram')
    plt.xlabel('Cluster')
    plt.ylabel('Distance')
    plt.show()

# Example dataset
data = np.array([
    [0.4, 0.53], [0.22, 0.38], [0.35, 0.32],
    [0.26, 0.19], [0.08, 0.41], [0.45, 0.30]
])

# Perform clustering
clusters, dendrogram_steps = single_linkage_clustering(data)

# Plot the dendrogram
plot_dendrogram(dendrogram_steps)
