"""
Applicability rules based on metadata.csv
"""

import csv
from typing import Dict, Set, List, Any, Optional


def load_applicability_rules(metadata_path: str = "metadata.csv") -> Dict[str, Dict]:
    """
    Load applicability rules from metadata.csv

    Returns:
        Dict mapping attribute -> {
            'roles': List[str],  # applicable roles
            'custom': str,  # custom condition attribute (e.g., 'reception', 'target')
            'run_pass': str  # 'RUN', 'PASS', or None
        }
    """
    rules = {}

    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            attr = row['target_id']

            # Parse applicable roles
            roles_str = row.get('applicable_roles', '')
            roles = []
            if roles_str and roles_str.strip():
                # Parse string like "['RUN DEFENSE', 'COVERAGE', 'PASS RUSH']"
                roles_str = roles_str.strip("[]'\"")
                if roles_str:
                    roles = [r.strip().strip("'\"") for r in roles_str.split(',')]

            # Get custom applicability condition
            custom = row.get('applicable_custom', '').strip()

            # Get run/pass applicability
            run_pass = row.get('applicable_run_pass', '').strip()

            rules[attr] = {
                'roles': roles,
                'custom': custom if custom else None,
                'run_pass': run_pass if run_pass else None
            }

    return rules


def is_applicable(
    attribute: str,
    batch_data: Dict[str, Any],
    applicability_rules: Dict[str, Dict]
) -> bool:
    """
    Check if an attribute is applicable based on batch predictions.

    Uses batch model's own predictions to determine applicability.

    Args:
        attribute: Attribute name (without prefix)
        batch_data: All batch predictions for this record
        applicability_rules: Rules loaded from metadata

    Returns:
        True if attribute should be evaluated
    """
    # If no rule exists, assume applicable
    if attribute not in applicability_rules:
        return True

    rule = applicability_rules[attribute]

    # Check role-based applicability
    if rule['roles']:
        batch_role = batch_data.get('role')
        if batch_role and batch_role not in rule['roles']:
            return False

    # Check custom condition (e.g., reception, target)
    if rule['custom']:
        custom_attr = rule['custom']
        custom_value = batch_data.get(custom_attr)

        # Custom condition must be True
        if custom_value is not True and custom_value != 'true':
            return False

    # Check run/pass applicability
    if rule['run_pass']:
        # Would need play_type from play-level data
        # For now, we can't filter on this at player_play level
        pass

    return True


if __name__ == "__main__":
    # Test loading rules
    rules = load_applicability_rules()

    print("\n" + "="*80)
    print("APPLICABILITY RULES")
    print("="*80 + "\n")

    # Show some interesting rules
    test_attrs = ['rec_yards', 'drop', 'reception', 'rush_yards', 'tackle', 'gap_assignment']

    for attr in test_attrs:
        if attr in rules:
            rule = rules[attr]
            print(f"{attr}:")
            if rule['roles']:
                print(f"  Roles: {rule['roles']}")
            if rule['custom']:
                print(f"  Custom: {rule['custom']} must be True")
            if rule['run_pass']:
                print(f"  Run/Pass: {rule['run_pass']}")
            if not rule['roles'] and not rule['custom'] and not rule['run_pass']:
                print(f"  No restrictions (always applicable)")
            print()
