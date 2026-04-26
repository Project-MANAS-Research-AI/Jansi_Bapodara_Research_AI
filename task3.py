import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN, HDBSCAN

np.random.seed(42)

print("="*50)
print("TASK C: SPATIAL-SEMANTIC BRIDGE")
print("="*50)

# ============================================================
# PART 1: SVM with α Extraction (Representer Theorem)
# ============================================================
print("\n[1] SVM α-EXTRACTOR")

X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.5, random_state=42)
y = np.where(y == 0, -1, 1)

svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X, y)

alpha_y = svm.dual_coef_[0]
support_vectors = svm.support_vectors_
b = svm.intercept_[0]

if svm.gamma == 'scale':
    gamma = 1.0 / (X.shape[1] * X.var())
else:
    gamma = svm.gamma

def rbf_kernel(x1, x2):
    return np.exp(-gamma * np.sum((np.asarray(x1) - np.asarray(x2))**2))

def decision_function(x):
    return np.sum(alpha_y * np.array([rbf_kernel(sv, x) for sv in support_vectors])) + b

x_test = X[0]
score = decision_function(x_test)
print(f"Test point decision score: {score:.4f}")
print(f"Support vectors: {len(support_vectors)}")

# Attribution
attributions = []
for j, sv in enumerate(support_vectors):
    k = rbf_kernel(sv, x_test)
    contrib = alpha_y[j] * k
    attributions.append({'sv': sv, 'contribution': contrib})

attributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
print(f"Top 5 contributions: {[round(a['contribution'], 3) for a in attributions[:5]]}")

# Counterfactual
original_sign = np.sign(score)
class_minus = X[y == -1].mean(axis=0)
class_plus = X[y == 1].mean(axis=0)
direction = (class_minus - x_test) if original_sign > 0 else (class_plus - x_test)
direction = direction / np.linalg.norm(direction)

x_perturbed = x_test.copy()
for i in range(100):
    if np.sign(decision_function(x_perturbed)) != original_sign:
        break
    x_perturbed = x_perturbed + 0.1 * direction

counterfactual = {
    'original_point': x_test.tolist(),
    'perturbed_point': x_perturbed.tolist(),
    'perturbation': (x_perturbed - x_test).tolist(),
    'perturbation_norm': float(np.linalg.norm(x_perturbed - x_test)),
    'original_decision': float(score),
    'perturbed_decision': float(decision_function(x_perturbed))
}

with open('counterfactual.json', 'w') as f:
    json.dump(counterfactual, f, indent=2)
print(f"Counterfactual: norm={counterfactual['perturbation_norm']:.4f}")
print("Saved: counterfactual.json")

# ============================================================
# PART 2: HDBSCAN vs DBSCAN (Stability Analysis)
# ============================================================
print("\n[2] CLUSTER STABILITY ANALYSIS")

dense_data, _ = make_blobs(n_samples=200, centers=1, cluster_std=0.5, center_box=(-2, -2), random_state=42)
sparse_data, _ = make_blobs(n_samples=80, centers=1, cluster_std=2.5, center_box=(4, 4), random_state=42)
noise = np.random.uniform(-6, 8, (50, 2))
X_cluster = np.vstack([dense_data, sparse_data, noise])

hdbscan = HDBSCAN(min_cluster_size=5)
labels_hdb = hdbscan.fit_predict(X_cluster)

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X_cluster)

hdb_clusters = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
db_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)

print(f"HDBSCAN: {hdb_clusters} clusters, {sum(labels_hdb==-1)} noise")
print(f"DBSCAN (eps=0.5): {db_clusters} clusters, {sum(labels_db==-1)} noise")

# Epsilon sweep
eps_range = np.linspace(0.1, 2.0, 30)
cluster_counts = []
for eps in eps_range:
    db = DBSCAN(eps=eps, min_samples=5)
    lbls = db.fit_predict(X_cluster)
    n = len(set(lbls)) - (1 if -1 in lbls else 0)
    cluster_counts.append(n)

# Persistence report
report = {
    'stability_metrics': {
        'n_clusters': int(hdb_clusters),
        'noise_points': int(sum(labels_hdb == -1)),
        'probabilities': hdbscan.probabilities_.tolist()
    },
    'epsilon_sweep': [{'eps': float(eps), 'clusters': int(nc)} 
                      for eps, nc in zip(eps_range, cluster_counts)]
}

with open('persistence_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print("Saved: persistence_report.json")

# ============================================================
# PART 3: Visualization
# ============================================================
print("\n[3] VISUALIZATION")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# SVM decision boundary
ax = axes[0, 0]
h = 0.1
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = np.array([decision_function(x) for x in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5, s=20)
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=80, 
           facecolors='none', edgecolors='green', linewidths=1.5)
ax.set_title('SVM: Decision Boundary + Support Vectors')

# Counterfactual
ax = axes[0, 1]
ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.3, s=20)
ax.scatter(x_test[0], x_test[1], c='blue', s=150, marker='o', 
           edgecolors='black', linewidths=2, label='Original')
ax.scatter(x_perturbed[0], x_perturbed[1], c='red', s=150, marker='*', 
           edgecolors='black', linewidths=2, label='Counterfactual')
ax.arrow(x_test[0], x_test[1], x_perturbed[0]-x_test[0], x_perturbed[1]-x_test[1],
         head_width=0.15, head_length=0.1, fc='purple', ec='purple')
ax.set_title('Counterfactual Perturbation')
ax.legend()

# HDBSCAN vs DBSCAN
ax = axes[1, 0]
colors_hdb = ['blue' if l==0 else 'red' if l==1 else 'gray' for l in labels_hdb]
ax.scatter(X_cluster[:, 0], X_cluster[:, 1], c=colors_hdb, alpha=0.6, s=20)
ax.set_title(f'HDBSCAN ({hdb_clusters} clusters, {sum(labels_hdb==-1)} noise)')

ax = axes[1, 1]
ax.plot(eps_range, cluster_counts, 'b-', linewidth=2)
ax.axvline(x=0.5, color='r', linestyle='--', label='eps=0.5')
ax.set_xlabel('Epsilon')
ax.set_ylabel('Number of Clusters')
ax.set_title('DBSCAN: Clusters vs Epsilon')
ax.legend()

plt.tight_layout()
plt.savefig('spatial_semantic_bridge.png', dpi=150)
print("Saved: spatial_semantic_bridge.png")

print("\n" + "="*50)
print("COMPLETE!")
print("="*50)