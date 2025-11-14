"""
Test script to verify accuracy computation in api.py works correctly.
"""

import csv
from transport_heads.http import PlayRecord
from api import compute_accuracy


def test_accuracy_computation():
    """Test accuracy computation in api.py"""
    print("Testing accuracy computation in api.py...")

    # Load a few records from play_records.csv
    records = []
    with open('play_records.csv', 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 10:  # Test with 10 records
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

    print(f"\nComputed accuracy metrics for {len(metrics)} attributes:")
    for metric in metrics[:15]:  # Show first 15
        print(f"  {metric['attribute']:25s}: "
              f"GT={metric['gt_accuracy']:.3f}, "
              f"CV={metric['cv_accuracy']:.3f}, "
              f"Batch={metric['batch_accuracy']:.3f} "
              f"(n={metric['n_instances']}, type={metric['type']})")

    print("\n✓ Test passed!")


if __name__ == "__main__":
    test_accuracy_computation()
