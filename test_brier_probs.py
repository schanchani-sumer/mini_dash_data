"""
Quick test to verify probability-based Brier Score calculation
"""
import ast
from eval_metrics import calculate_brier_skill_score

# Test binary classification with probabilities
y_true_binary = [True, False, True, True, False]
y_pred_binary = [True, False, True, False, False]
y_pred_probs_binary = [
    {'true': 0.9, 'false': 0.1},
    {'true': 0.2, 'false': 0.8},
    {'true': 0.85, 'false': 0.15},
    {'true': 0.4, 'false': 0.6},
    {'true': 0.3, 'false': 0.7}
]

print("Testing Binary Classification:")
print("=" * 50)

# Without probabilities (hard predictions)
bss_hard = calculate_brier_skill_score(y_true_binary, y_pred_binary, is_binary=True)
print(f"BSS (hard predictions): {bss_hard:.4f}")

# With probabilities
bss_prob = calculate_brier_skill_score(
    y_true_binary, y_pred_binary, is_binary=True,
    y_pred_probs=y_pred_probs_binary
)
print(f"BSS (with probabilities): {bss_prob:.4f}")

print("\nExpected: Probabilistic BSS should be different (likely lower) than hard prediction BSS")
print("This shows we're using the actual probability distributions!")

# Test categorical
y_true_cat = ['run', 'pass', 'run', 'pass', 'run']
y_pred_cat = ['run', 'pass', 'pass', 'pass', 'run']
y_pred_probs_cat = [
    {'run': 0.7, 'pass': 0.3},
    {'run': 0.2, 'pass': 0.8},
    {'run': 0.45, 'pass': 0.55},
    {'run': 0.1, 'pass': 0.9},
    {'run': 0.85, 'pass': 0.15}
]

print("\n" + "=" * 50)
print("Testing Categorical Classification:")
print("=" * 50)

# Without probabilities
bss_cat_hard = calculate_brier_skill_score(y_true_cat, y_pred_cat, is_binary=False)
print(f"BSS (hard predictions): {bss_cat_hard:.4f}")

# With probabilities
bss_cat_prob = calculate_brier_skill_score(
    y_true_cat, y_pred_cat, is_binary=False,
    y_pred_probs=y_pred_probs_cat
)
print(f"BSS (with probabilities): {bss_cat_prob:.4f}")

print("\nTest CSV probability parsing:")
print("=" * 50)
csv_prob_str = "{'true': 0.27048352360725403, 'false': 0.729516476392746}"
parsed = ast.literal_eval(csv_prob_str)
print(f"Original: {csv_prob_str}")
print(f"Parsed: {parsed}")
print(f"Type: {type(parsed)}")
print(f"Value for 'true': {parsed['true']}")
