"""
Debug script to see what's happening with yards_gained accuracy.
"""

import csv

def debug_yards_gained():
    """Debug yards_gained values"""
    print("Debugging yards_gained values...\n")

    records = []
    with open('play_records.csv', 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 20:  # Check first 20 records
                break

            label = row.get('label_yards_gained')
            gt = row.get('gt_yards_gained')
            cv = row.get('cv_yards_gained')
            batch = row.get('batch_yards_gained')

            # Only show if all are present
            if label and gt and cv and batch:
                try:
                    label_val = float(label)
                    gt_val = float(gt)
                    cv_val = float(cv)
                    batch_val = float(batch)

                    # Calculate 10% tolerance
                    if abs(label_val) < 1e-6:
                        gt_match = abs(gt_val) < 0.1
                        cv_match = abs(cv_val) < 0.1
                        batch_match = abs(batch_val) < 0.1
                    else:
                        gt_match = abs(gt_val - label_val) / abs(label_val) <= 0.10
                        cv_match = abs(cv_val - label_val) / abs(label_val) <= 0.10
                        batch_match = abs(batch_val - label_val) / abs(label_val) <= 0.10

                    print(f"Record {i+1}:")
                    print(f"  Label: {label_val:.2f}")
                    print(f"  GT:    {gt_val:.2f}  (diff: {abs(gt_val - label_val):.2f}, {abs(gt_val - label_val) / max(abs(label_val), 0.01) * 100:.1f}%) {'✓' if gt_match else '✗'}")
                    print(f"  CV:    {cv_val:.2f}  (diff: {abs(cv_val - label_val):.2f}, {abs(cv_val - label_val) / max(abs(label_val), 0.01) * 100:.1f}%) {'✓' if cv_match else '✗'}")
                    print(f"  Batch: {batch_val:.2f}  (diff: {abs(batch_val - label_val):.2f}, {abs(batch_val - label_val) / max(abs(label_val), 0.01) * 100:.1f}%) {'✓' if batch_match else '✗'}")
                    print()
                except ValueError:
                    print(f"Record {i+1}: Could not convert to float - label={label}, gt={gt}, cv={cv}, batch={batch}\n")


if __name__ == "__main__":
    debug_yards_gained()
