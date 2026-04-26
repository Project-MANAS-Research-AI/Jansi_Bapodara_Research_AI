import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Load iris dataset (binary classification: classes 0 and 1)
iris = load_iris()
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

# Train SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X, y)

# Get model parameters
alpha_y = svm.dual_coef_[0]  # αᵢ × yᵢ (pre-multiplied)
support_vectors = svm.support_vectors_
b = svm.intercept_[0]

# Compute actual gamma value (sklearn's 'scale' = 1 / (n_features * X.var()))
if svm.gamma == 'scale':
    gamma = 1.0 / (X.shape[1] * X.var())
else:
    gamma = svm.gamma

print(f"Number of support vectors: {len(support_vectors)}")
print(f"Gamma (computed): {gamma:.6f}")
print(f"Bias term (b): {b:.6f}")

# Define RBF kernel
def rbf_kernel(x1, x2, gamma):
    """K(x1, x2) = exp(-gamma * ||x1 - x2||²)"""
    diff = np.asarray(x1) - np.asarray(x2)
    return np.exp(-gamma * np.dot(diff, diff))

# Decision function using Representer Theorem
def decision_function(x_test, alpha_y, support_vectors, b, gamma):
    """
    f(x) = Σᵢ (αᵢ × yᵢ) × K(xᵢ, x) + b
    
    This is the core Representer Theorem formulation:
    - Each support vector contributes based on its similarity (kernel) to x
    - The sign of αᵢ×yᵢ determines if it pushes toward +1 or -1
    - The magnitude |αᵢ×yᵢ| determines the strength of contribution
    """
    kernel_sum = np.sum(alpha_y * np.array([rbf_kernel(sv, x_test, gamma) for sv in support_vectors]))
    return kernel_sum + b

# Test on a sample point
x_test = X[10]  # Any point from dataset
score = decision_function(x_test, alpha_y, support_vectors, b, gamma)
print(f"\nDecision score for test point: {score:.4f}")
print(f"Predicted class: {'1' if score > 0 else '0'}")

# ============================================================
# DECISION ATTRIBUTION: Which support vectors contributed most?
# ============================================================
# For each support vector: contribution_i = (αᵢ × yᵢ) × K(svᵢ, x_test)
# This shows exactly how much each SV pushes the decision

contributions = []
for j, sv in enumerate(support_vectors):
    kernel_val = rbf_kernel(sv, x_test, gamma)
    contrib = alpha_y[j] * kernel_val
    contributions.append({
        'sv_index': svm.support_[j],
        'sv': sv,
        'alpha_y': alpha_y[j],  # αᵢ × yᵢ
        'kernel': kernel_val,   # K(svᵢ, x_test)
        'contribution': contrib # (αᵢ × yᵢ) × K
    })

# Sort by absolute contribution
contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

print("\n" + "="*70)
print("TOP 10 SUPPORT VECTORS BY CONTRIBUTION TO DECISION")
print("="*70)
print(f"{'Rank':<5} {'αᵢ×yᵢ':<10} {'K(sv,x)':<10} {'Contribution':<15} {'SV (first 2 dims)'}")
print("-"*70)

for i, c in enumerate(contributions[:10]):
    print(f"{i+1:<5} {c['alpha_y']:<10.4f} {c['kernel']:<10.4f} {c['contribution']:<15.4f} "
          f"[{c['sv'][0]:.2f}, {c['sv'][1]:.2f}, ...]")

# Verify reconstruction
total_contrib = sum(c['contribution'] for c in contributions)
print(f"\nVerification: Σ contributions + b = {total_contrib:.4f} + {b:.4f} = {total_contrib + b:.4f}")
print(f"Actual decision function: {score:.4f}")

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Data with support vectors
ax1 = axes[0]
colors = ['blue' if label == 0 else 'red' for label in y]
ax1.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5, s=30, label='Training data')
ax1.scatter(support_vectors[:, 0], support_vectors[:, 1], s=150, 
            facecolors='none', edgecolors='green', linewidths=2, label='Support vectors')
ax1.scatter(x_test[0], x_test[1], c='yellow', s=200, marker='*', 
            edgecolors='black', linewidths=2, label='Test point')
ax1.set_xlabel('Sepal length')
ax1.set_ylabel('Sepal width')
ax1.set_title('SVM with RBF Kernel: Support Vectors')
ax1.legend()

# Plot 2: Contribution bar chart
ax2 = axes[1]
top_contributors = contributions[:15]
labels = [f"SV {c['sv_index']}" for c in top_contributors]
values = [c['contribution'] for c in top_contributors]
colors_bar = ['green' if v > 0 else 'red' for v in values]

bars = ax2.barh(range(len(labels)), values, color=colors_bar, alpha=0.7)
ax2.set_yticks(range(len(labels)))
ax2.set_yticklabels(labels)
ax2.set_xlabel('Contribution to Decision Score')
ax2.set_title('Support Vector Contributions (Representer Theorem)')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax2.invert_yaxis()

# Add value labels on bars
for bar, val in zip(bars, values):
    ax2.text(val + 0.01 * np.sign(val), bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('representer_attribution.png', dpi=150)
plt.show()

print("\nPlot saved to 'representer_attribution.png'")
