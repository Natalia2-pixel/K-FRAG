"""Audit K-FRAG/WAM/InvisMark matched crop metadata and recovery counts.

Usage (run at repository root):
    python tests/audit_matched_crop_baselines.py

Uses only Python standard library. Does not execute neural inference.
"""
from collections import Counter
from csv import DictReader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results' / 'matched_crop_baselines'
AREAS = (25, 30, 40, 50, 60, 75, 80, 95)
GEOMETRY = ('crop_x', 'crop_y', 'crop_width', 'crop_height')


def load(name):
    with (DATA / name).open(newline='', encoding='utf-8-sig') as stream:
        return list(DictReader(stream))


def key(row):
    return row['image'], int(float(row['nominal_area_pct']))


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    manifest = load('kfrag_v415_crop_manifest.csv')
    check(len(manifest) == 800, 'Manifest must have 800 crop rows')
    records = {key(row): row for row in manifest}
    check(len(records) == 800, 'Duplicate crop configuration in manifest')
    check(len({r['image'] for r in manifest}) == 100, 'Expected 100 manifest images')
    for area in AREAS:
        check(sum(key(r)[1] == area for r in manifest) == 100,
              f'Expected 100 manifest rows for area {area}')
    for r in manifest:
        check(int(r['crop_x']) >= 0 and int(r['crop_y']) >= 0,
              f'Negative crop coordinate: {key(r)}')
        check(int(r['crop_x']) + int(r['crop_width']) <= 256 and
              int(r['crop_y']) + int(r['crop_height']) <= 256,
              f'Out-of-bounds crop: {key(r)}')
        actual = 100 * int(r['crop_width']) * int(r['crop_height']) / (256 * 256)
        check(abs(actual - float(r['actual_area_pct'])) < 1e-8,
              f'Area mismatch: {key(r)}')

    for filename, bits, name in (
        ('wam_kfrag_matched_results.csv', 32, 'WAM'),
        ('invismark_matched_results.csv', 100, 'InvisMark'),
    ):
        rows = load(filename)
        crop = [r for r in rows if r['condition'] != 'INTACT']
        intact = [r for r in rows if r['condition'] == 'INTACT']
        check(len(rows) == 900 and len(crop) == 800 and len(intact) == 100,
              f'{name}: wrong crop/intact counts')
        check(len({key(r) for r in crop}) == 800,
              f'{name}: duplicate image/area crop keys')
        check({key(r) for r in crop} == set(records),
              f'{name}: missing or extra crop configurations')
        success = Counter()
        for r in crop:
            k = key(r)
            m = records[k]
            check(r['condition'] == f'CROP_{k[1]}',
                  f'{name}: condition mismatch at {k}')
            check(all(int(r[field]) == int(m[field]) for field in GEOMETRY),
                  f'{name}: geometry mismatch at {k}')
            check(abs(float(r['actual_area_pct']) - float(m['actual_area_pct'])) < 1e-8,
                  f'{name}: actual area mismatch at {k}')
            check(abs(float(r['bit_accuracy_pct']) -
                      100 * int(r['correct_bits']) / bits) < 1e-8,
                  f'{name}: bit accuracy mismatch at {k}')
            check((r['exact_recovery'] == 'True') == (int(r['correct_bits']) == bits),
                  f'{name}: exact recovery mismatch at {k}')
            success[k[1]] += int(r['exact_recovery'] == 'True')
        print(f'{name}: PASS (800/800 crop geometry matches, 100 intact controls)')
        print('  ' + ', '.join(f'{a}%: {success[a]}/100' for a in AREAS))
    print('NOTE: This audits recorded CSVs; it does not rerun neural inference or K-FRAG verification.')


if __name__ == '__main__':
    main()
