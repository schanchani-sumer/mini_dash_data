"""
Simple test script to verify accuracy_api.py works correctly.
"""

import csv
from transport_heads.accuracy_http import PlayRecord
from accuracy_api import compute_accuracy


def test_accuracy_computation():
    """Test basic accuracy computation"""
    print("Testing accuracy computation...")

    # Load a few records from play_records.csv
    records = []
    with open('play_records.csv', 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 5:  # Just test with 5 records
                break

            # Convert string values to appropriate types
            data = {}
            for key, value in row.items():
                if value == '':
                    data[key] = None
                elif value.lower() == 'true':
                    data[key] = True
                elif value.lower() == 'false':
                    data[key] = False
                else:
                    try:
                        data[key] = float(value)
                    except (ValueError, TypeError):
                        data[key] = value

            record = PlayRecord(
                game_id=row['game_id'],
                play_id=row['play_id'],
                data=data
            )
            records.append(record)

    print(f"Loaded {len(records)} records")

    # Compute accuracy metrics
    metrics = compute_accuracy(records)

    print(f"\nComputed metrics for {len(metrics)} attributes:")
    for metric in metrics[:10]:  # Show first 10
        print(f"  {metric['attribute']}: "
              f"GT={metric['gt_accuracy']:.3f}, "
              f"CV={metric['cv_accuracy']:.3f}, "
              f"Batch={metric['batch_accuracy']:.3f} "
              f"(n={metric['n_instances']})")

    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_accuracy_computation()
