"""Audit complete AF-priority labels and folds without training or modifying runs."""
import argparse
import json
import tempfile
from cpsc2018.data import make_splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--labels', default='')
    p.add_argument('--reference-csv')
    p.add_argument('--expected-af-count', type=int, default=1221)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--split-mode', choices=('folds', 'stratified'), default='folds')
    a = p.parse_args()
    with tempfile.TemporaryDirectory(prefix='ovdl_af_priority_v2_check_') as temporary:
        _, _, manifest = make_splits(a.data_dir, a.labels, temporary, a.split_seed,
                                    a.split_mode, a.expected_af_count, a.reference_csv)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
