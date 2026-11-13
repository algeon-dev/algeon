#!/usr/bin/env python3
"""Measure the Parquet layout of a TPC fixture directory and write it as layout.json.

The directory must hold the `_generation.json` manifest that fixture generation wrote.
Measurements come from the file footers, OffsetIndex structures and page headers; a
mismatch with the manifest is an error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parquet_layout import describe_layout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_dir", type=Path, help="Directory with one Parquet file per table and _generation.json")
    parser.add_argument("--out", type=Path, required=True, help="Path of the layout.json to write")
    args = parser.parse_args()
    layout = describe_layout(args.parquet_dir)
    args.out.write_text(json.dumps(layout, indent=2) + "\n")


if __name__ == "__main__":
    main()
