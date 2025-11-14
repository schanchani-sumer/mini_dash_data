"""
Test script to verify yards_gained accuracy with 1 yard tolerance.
"""

import csv
from transport_heads.http import PlayRecord
from api import compute_accuracy


def test_yards_accuracy():
    """Test yards_gained accuracy with 1 yard tolerance"""
    print("Testing yards_gained accuracy with 1 yard tolerance...\n")

    # Load records from play_records.csv
    records = []
    with open('play_records.csv', 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 50:  # Test with 50 records
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

    # Find yards_gained metric
    yards_gained = None
    yards_after_contact = None
    for metric in metrics:
        if metric['attribute'] == 'yards_gained':
            yards_gained = metric
        elif metric['attribute'] == 'yards_after_contact':
            yards_after_contact = metric

    print("\n=== Yard-based Metrics (1 yard tolerance) ===")
    if yards_gained:
        print(f"yards_gained:")
        print(f"  GT:    {yards_gained['gt_accuracy']*100:.1f}%")
        print(f"  CV:    {yards_gained['cv_accuracy']*100:.1f}%")
        print(f"  Batch: {yards_gained['batch_accuracy']*100:.1f}%")
        print(f"  n={yards_gained['n_instances']}")

    if yards_after_contact:
        print(f"\nyards_after_contact:")
        print(f"  GT:    {yards_after_contact['gt_accuracy']*100:.1f}%")
        print(f"  CV:    {yards_after_contact['cv_accuracy']*100:.1f}%")
        print(f"  Batch: {yards_after_contact['batch_accuracy']*100:.1f}%")
        print(f"  n={yards_after_contact['n_instances']}")

    print("\n=== Other Continuous Metrics (10% tolerance) ===")
    for metric in metrics[:10]:
        if metric['type'] == 'continuous' and 'yard' not in metric['attribute']:
            print(f"{metric['attribute']}:")
            print(f"  GT:    {metric['gt_accuracy']*100:.1f}%")
            print(f"  CV:    {metric['cv_accuracy']*100:.1f}%")
            print(f"  Batch: {metric['batch_accuracy']*100:.1f}%")
            print(f"  n={metric['n_instances']}")
            break

    print("\n✓ Test completed!")


if __name__ == "__main__":
    test_yards_accuracy()
