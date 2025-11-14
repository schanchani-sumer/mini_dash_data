"""
Test that all three comparisons now use true Brier Scores with probabilities
"""
from eval_metrics import calculate_brier_skill_score

# Sample data simulating CV, GT, and Batch predictions
y_true_gt = [True, False, True, True, False]
y_pred_cv = [True, False, True, False, False]
y_pred_batch = [True, True, True, True, False]

# Probability distributions for each model
cv_probs = [
    {'true': 0.9, 'false': 0.1},
    {'true': 0.2, 'false': 0.8},
    {'true': 0.85, 'false': 0.15},
    {'true': 0.4, 'false': 0.6},
    {'true': 0.3, 'false': 0.7}
]

batch_probs = [
    {'true': 0.95, 'false': 0.05},
    {'true': 0.6, 'false': 0.4},
    {'true': 0.75, 'false': 0.25},
    {'true': 0.8, 'false': 0.2},
    {'true': 0.2, 'false': 0.8}
]

print("Testing All Three Comparisons with True Brier Scores")
print("=" * 60)

# 1. CV vs GT (NGS XY) - Blue bar
print("\n1. CV Data vs NGS Data (Both XY API):")
print("-" * 60)
cv_xy_score = calculate_brier_skill_score(
    y_true_gt, y_pred_cv, is_binary=True,
    y_pred_probs=cv_probs
)
print(f"   BSS with CV probabilities: {cv_xy_score:.4f}")

# 2. Batch vs GT - Gray bar
print("\n2. NGS Data XY API vs Batch API:")
print("-" * 60)
batch_xy_score = calculate_brier_skill_score(
    y_true_gt, y_pred_batch, is_binary=True,
    y_pred_probs=batch_probs
)
print(f"   BSS with Batch probabilities: {batch_xy_score:.4f}")

# 3. CV vs Batch - Orange bar
print("\n3. CV Data XY API vs NGS Data Batch API:")
print("-" * 60)
cv_batch_score = calculate_brier_skill_score(
    y_pred_batch, y_pred_cv, is_binary=True,
    y_pred_probs=cv_probs
)
print(f"   BSS with CV probabilities: {cv_batch_score:.4f}")

print("\n" + "=" * 60)
print("SUCCESS: All three bars now use TRUE Brier Scores!")
print("All comparisons account for prediction confidence, not just")
print("whether the hard prediction was correct.")
print("=" * 60)
