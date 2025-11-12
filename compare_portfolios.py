#!/usr/bin/env python3
import sys

def parse_file(path):
    """
    Parse a file of lines like 'KEY: value' into a dict {KEY: float(value)}.
    Lines that don't match are ignored.
    """
    d = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(':', 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            try:
                val = float(parts[1].strip())
            except ValueError:
                continue
            d[key] = val
    return d

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} fileA fileB")
        sys.exit(1)

    fileA, fileB = sys.argv[1], sys.argv[2]
    A = parse_file(fileA)
    B = parse_file(fileB)

    # Collect all keys
    all_keys = set(A.keys()) | set(B.keys())

    # Build list of (key, B_value, diff)
    results = []
    for key in all_keys:
        a_val = A.get(key, 0.0)
        b_val = B.get(key, 0.0)
        diff  = b_val - a_val
        results.append((key, b_val, diff))

    # Sort by absolute B_value, descending
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    # Print
    for key, b_val, diff in results:
        # + sign for positive diffs
        sign_str = f"{diff:+.10f}"
        print(f"{key}: {b_val:.15f} ({sign_str})")

if __name__ == "__main__":
    main()