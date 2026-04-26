import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GENERATE MIXED-DENSITY DATASET
# ============================================================
np.random.seed(42)

# Dense cluster
dense_data, _ = make_blobs(n_samples=200, centers=1, cluster_std=0.5, 
                           center_box=(-2, -2), random_state=42)

# Sparse cluster (lower density - harder to detect)
sparse_data, _ = make_blobs(n_samples=80, centers=1, cluster_std=2.5, 
                            center_box=(4, 4), random_state=42)

# Noise points
noise = np.random.uniform(-6, 8, (50, 2))

X = np.vstack([dense_data, sparse_data, noise])
true_labels = np.array([0]*200 + [1]*80 + [-1]*50)

print(f"Dataset: {len(X)} points, 2 clusters + noise")
print(f"Dense cluster: 200 points, std=0.5")
print(f"Sparse cluster: 80 points, std=2.5")

# ============================================================
# RUN DBSCAN AND HDBSCAN
# ============================================================
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)

hdbscan = HDBSCAN(min_cluster_size=5)
labels_hdbscan = hdbscan.fit_predict(X)

print(f"\nDBSCAN (eps=0.5): {len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)} clusters")
print(f"HDBSCAN: {len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)} clusters")
print(f"HDBSCAN noise points: {sum(labels_hdbscan == -1)}")

# ============================================================
# EXTRACT HDBSCAN HIERARCHY AND STABILITY
# ============================================================

# Get cluster probabilities (strength/membership probability)
cluster_probs = hdbscan.probabilities_

print("\n" + "="*60)
print("HDBSCAN CLUSTER STABILITY ANALYSIS")
print("="*60)

for i in range(max(labels_hdbscan) + 1):
    mask = labels_hdbscan == i
    avg_prob = cluster_probs[mask].mean()
    print(f"Cluster {i}: Avg probability = {avg_prob:.4f}")

# ============================================================
# EXTRACT CONDENSED TREE (HIERARCHY)
# ============================================================
condensed_tree = hdbscan.condensed_tree_

# Handle different sklearn HDBSCAN API versions
try:
    cluster_tree = condensed_tree._cluster_tree
except AttributeError:
    cluster_tree = condensed_tree.tree.to_pandas()

print("\n" + "="*60)
print("CONDENSED TREE ANALYSIS (Lambda values)")
print("="*60)

# Extract lambda values from the tree
lambda_values = cluster_tree['lambda_val'].values
parent = cluster_tree['parent'].values
child = cluster_tree['child'].values
size = cluster_tree['size'].values

print(f"Lambda range in hierarchy: {lambda_values.min():.4f} to {lambda_values.max():.4f}")

# Find cluster entry/exit points
cluster_creation_lambda = {}
for i in range(len(cluster_tree)):
    if parent[i] == 0:  # Root cluster
        cluster_id = child[i]
        cluster_creation_lambda[cluster_id] = lambda_values[i]

print(f"\nCluster creation lambdas: {cluster_creation_lambda}")

# ============================================================
# MAP HDBSCAN LAMBDA TO DBSCAN EPSILON
# ============================================================
lambda_min = lambda_values.min()
lambda_max = lambda_values.max()
epsilon_min = 1 / lambda_max
epsilon_max = 1 / lambda_min

print("\n" + "="*60)
print("MAPPING HDBSCAN λ TO DBSCAN ε")
print("="*60)
print(f"HDBSCAN λ range: [{lambda_min:.4f}, {lambda_max:.4f}]")
print(f"DBSCAN ε range: [{epsilon_min:.4f}, {epsilon_max:.4f}]")

for cluster_id, entry_lambda in cluster_creation_lambda.items():
    noise_epsilon = 1 / entry_lambda
    print(f"\nCluster {cluster_id}:")
    print(f"  Entry λ = {entry_lambda:.4f} → DBSCAN ε = {noise_epsilon:.4f}")
    print(f"  Below ε = {noise_epsilon:.4f}, cluster becomes noise in DBSCAN")

# ============================================================
# SWEEP DBSCAN ACROSS EPSILON RANGE
# ============================================================
eps_range = np.linspace(0.1, 2.0, 50)
n_clusters = []
noise_counts = []

for eps in eps_range:
    db = DBSCAN(eps=eps, min_samples=5)
    labels = db.fit_predict(X)
    n_clusters.append(len(set(labels)) - (1 if -1 in labels else 0))
    noise_counts.append(sum(labels == -1))

hdbscan_clusters = set(labels_hdbscan[labels_hdbscan >= 0])
sparse_cluster_eps = None

for i, (eps, n_clust) in enumerate(zip(eps_range, n_clusters)):
    if n_clust >= len(hdbscan_clusters):
        sparse_cluster_eps = eps
        break

print(f"\nSparse cluster emerges in DBSCAN at ε ≈ {sparse_cluster_eps:.2f}")

# ============================================================
# VISUALIZATION
# ============================================================
fig = plt.figure(figsize=(16, 12))

# Plot 1: Original Data with True Labels
ax1 = fig.add_subplot(2, 3, 1)
colors = ['blue' if l == 0 else 'red' if l == 1 else 'gray' for l in true_labels]
ax1.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=20)
ax1.set_title('Ground Truth (Mixed Density)')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')

# Plot 2: DBSCAN Results
ax2 = fig.add_subplot(2, 3, 2)
colors_db = ['blue' if l == 0 else 'red' if l == 1 else 'gray' for l in labels_dbscan]
ax2.scatter(X[:, 0], X[:, 1], c=colors_db, alpha=0.6, s=20)
ax2.set_title('DBSCAN (eps=0.5)\nMisses sparse cluster')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')

# Plot 3: HDBSCAN Results
ax3 = fig.add_subplot(2, 3, 3)
colors_hdb = ['blue' if l == 0 else 'red' if l == 1 else 'gray' for l in labels_hdbscan]
ax3.scatter(X[:, 0], X[:, 1], c=colors_hdb, alpha=0.6, s=20)
low_prob_mask = cluster_probs < 0.5
ax3.scatter(X[low_prob_mask, 0], X[low_prob_mask, 1], facecolors='none', 
            edgecolors='yellow', s=50, alpha=0.8, label='Low confidence')
ax3.set_title('HDBSCAN\nFinds both clusters')
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.legend()

# Plot 4: HDBSCAN Cluster Probabilities
ax4 = fig.add_subplot(2, 3, 4)
scatter = ax4.scatter(X[:, 0], X[:, 1], c=cluster_probs, cmap='viridis', 
                       alpha=0.7, s=30)
plt.colorbar(scatter, ax=ax4, label='Cluster Probability')
ax4.set_title('HDBSCAN: Cluster Membership Probability')
ax4.set_xlabel('Feature 1')
ax4.set_ylabel('Feature 2')

# Plot 5: DBSCAN Epsilon Sweep
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(eps_range, n_clusters, 'b-', linewidth=2, label='# Clusters')
ax5.axvline(x=0.5, color='r', linestyle='--', label='DBSCAN eps=0.5')
ax5.axvline(x=sparse_cluster_eps, color='g', linestyle='--', 
            label=f'Sparse cluster ε≈{sparse_cluster_eps:.2f}')
ax5.set_xlabel('Epsilon (ε)')
ax5.set_ylabel('Number of Clusters')
ax5.set_title('DBSCAN: Cluster Count vs Epsilon')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Lambda-Epsilon Mapping
ax6 = fig.add_subplot(2, 3, 6)
lambda_range = 1 / eps_range
ax6.plot(lambda_range, eps_range, 'purple', linewidth=2)
ax6.scatter([1/0.5], [0.5], color='red', s=100, zorder=5, label='DBSCAN eps=0.5')
for cluster_id, entry_l in cluster_creation_lambda.items():
    ax6.axvline(x=entry_l, color='green', linestyle='--', alpha=0.7, 
                label=f'Cluster {cluster_id} entry λ')
ax6.set_xlabel('Lambda (λ = 1/ε)')
ax6.set_ylabel('Epsilon (ε)')
ax6.set_title('HDBSCAN λ ↔ DBSCAN ε Mapping')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hdbscan_dbscan_analysis.png', dpi=150)
plt.show()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("SUMMARY: HDBSCAN STABILITY → DBSCAN EPSILON")
print("="*60)
print("""
KEY INSIGHTS:
1. HDBSCAN finds clusters at all density levels via hierarchy
2. Cluster STABILITY = persistence across lambda values
3. Higher stability → cluster exists over wider epsilon range
4. The sparse cluster has LOW stability → requires larger epsilon in DBSCAN
5. Mapping: ε = 1/λ where λ is the entry point in HDBSCAN hierarchy
""")