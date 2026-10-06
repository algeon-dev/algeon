#!/usr/bin/env python3
"""Prepare bounded research Parquet projections from the EMBER2024 raw inputs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from thrember.features import PEFeatureExtractor
from thrember.model import gather_feature_paths


FIXTURE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = FIXTURE_ROOT.parents[2]
RAW_DIR = FIXTURE_ROOT / "raw"
OUTPUT_DIR = WORKSPACE_ROOT / "target" / "ember2024-research"
PROVENANCE_PATH = WORKSPACE_ROOT / "docs" / "research" / "ember2024" / "results" / "preparation.json"

SPLITS = ("challenge", "test", "train")
ID_BASE = {"train": 0, "test": 10_000_000, "challenge": 20_000_000}
VECTOR_DIM = 2568
HISTOGRAM_START = 7
HISTOGRAM_STOP = 519
HISTOGRAM_DIM = HISTOGRAM_STOP - HISTOGRAM_START
METADATA_CHUNK_ROWS = 50_000
VECTOR_CHUNK_ROWS = 50_000
IMPORT_CHUNK_ROWS = 100_000
SQL_COMMIT_ROWS = 100_000

METADATA_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("source_id", pa.int64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("sha256", pa.string(), nullable=False),
        pa.field("file_type", pa.string(), nullable=False),
        pa.field("week_id", pa.int32(), nullable=False),
        pa.field("label", pa.int32(), nullable=False),
        pa.field("family", pa.string(), nullable=False),
        pa.field("family_confidence", pa.float32(), nullable=False),
        pa.field("detected", pa.int32(), nullable=False),
        pa.field("total_engines", pa.int32(), nullable=False),
        pa.field("first_submission_date", pa.int64(), nullable=False),
    ]
)
VECTOR_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("label", pa.int32(), nullable=False),
        pa.field("vector", pa.list_(pa.field("element", pa.float32(), nullable=False), VECTOR_DIM), nullable=False),
        pa.field(
            "histogram", pa.list_(pa.field("element", pa.float32(), nullable=False), HISTOGRAM_DIM), nullable=False
        ),
    ]
)
IMPORT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
    ]
)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def vcs_commit(name: str) -> str | None:
    try:
        direct_url = importlib.metadata.distribution(name).read_text("direct_url.json")
    except (importlib.metadata.PackageNotFoundError, OSError):
        return None
    if direct_url is None:
        return None
    try:
        return json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except (json.JSONDecodeError, AttributeError):
        return None


def split_memmap_shape(split: str) -> tuple[int, int]:
    x_path = RAW_DIR / f"X_{split}.dat"
    y_path = RAW_DIR / f"y_{split}.dat"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"missing vectorized files for {split}: {x_path}, {y_path}")

    y_bytes = y_path.stat().st_size
    if y_bytes % np.dtype(np.int32).itemsize:
        raise ValueError(f"{y_path} has a byte size that is not aligned to int32")
    rows = y_bytes // np.dtype(np.int32).itemsize
    x_bytes = x_path.stat().st_size
    row_bytes = rows * np.dtype(np.float32).itemsize
    if rows == 0 or x_bytes % row_bytes:
        raise ValueError(f"{x_path} does not have a whole number of float32 rows")
    dimension = x_bytes // row_bytes
    if dimension != VECTOR_DIM:
        raise ValueError(f"{split} memmap has {dimension} features, expected {VECTOR_DIM}")
    return rows, dimension


def parse_detection_ratio(value: Any, split: str, source_id: int) -> tuple[int, int]:
    if value is None:
        return 0, 0
    if not isinstance(value, str) or "/" not in value:
        raise ValueError(f"invalid detection_ratio at {split} source_id {source_id}: {value!r}")
    detected, total = value.split("/", 1)
    try:
        return int(detected), int(total)
    except ValueError as exc:
        raise ValueError(f"invalid detection_ratio at {split} source_id {source_id}: {value!r}") from exc


def raw_sha256(record: dict[str, Any], split: str, source_id: int) -> tuple[str, bytes]:
    digest_hex = record.get("sha256")
    if not isinstance(digest_hex, str) or len(digest_hex) != 64:
        raise ValueError(f"missing or malformed sha256 at {split} source_id {source_id}")
    try:
        digest = bytes.fromhex(digest_hex)
    except ValueError as exc:
        raise ValueError(f"malformed sha256 at {split} source_id {source_id}") from exc
    return digest_hex.lower(), digest


def is_selected(split: str, digest_hex: str) -> bool:
    return True


def write_metadata_chunk(writer: pq.ParquetWriter, columns: dict[str, list[Any]]) -> int:
    rows = len(columns["id"])
    if not rows:
        return 0
    arrays = [pa.array(columns[field.name], type=field.type) for field in METADATA_SCHEMA]
    writer.write_table(pa.Table.from_arrays(arrays, schema=METADATA_SCHEMA), row_group_size=rows)
    for values in columns.values():
        values.clear()
    return rows


def write_vector_chunk(writer: pq.ParquetWriter, rows: list[tuple[int, str, int, np.ndarray]]) -> int:
    if not rows:
        return 0
    ids = pa.array([row[0] for row in rows], type=pa.int64())
    splits = pa.array([row[1] for row in rows], type=pa.string())
    labels = pa.array([row[2] for row in rows], type=pa.int32())
    vectors = np.stack([row[3] for row in rows]).astype(np.float32, copy=False)
    vector_array = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()), type=VECTOR_SCHEMA.field("vector").type
    )
    histograms = vectors[:, HISTOGRAM_START:HISTOGRAM_STOP]
    histogram_array = pa.FixedSizeListArray.from_arrays(
        pa.array(histograms.reshape(-1), type=pa.float32()), type=VECTOR_SCHEMA.field("histogram").type
    )
    table = pa.Table.from_arrays([ids, splits, labels, vector_array, histogram_array], schema=VECTOR_SCHEMA)
    writer.write_table(table, row_group_size=len(rows))
    rows.clear()
    return table.num_rows


def write_import_chunk(writer: pq.ParquetWriter, rows: list[tuple[int, str, str]]) -> int:
    if not rows:
        return 0
    columns = list(zip(*rows))
    table = pa.Table.from_arrays(
        [
            pa.array(columns[0], type=pa.int64()),
            pa.array(columns[1], type=pa.string()),
            pa.array(columns[2], type=pa.string()),
        ],
        schema=IMPORT_SCHEMA,
    )
    writer.write_table(table, row_group_size=len(rows))
    rows.clear()
    return table.num_rows


def extract_import_symbols(imports: Any) -> list[str]:
    if imports is None:
        return []
    if not isinstance(imports, dict):
        raise ValueError(f"imports must be a JSON object or null, got {type(imports).__name__}")
    symbols: list[str] = []
    seen: set[str] = set()
    for dll, apis in imports.items():
        if apis is None:
            continue
        if isinstance(apis, str):
            api_names = [apis]
        elif isinstance(apis, list):
            api_names = apis
        else:
            raise ValueError(f"imports for {dll!r} must be a list or string")
        for api in api_names:
            symbol = f"{str(dll).lower()}!{api}"
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
    return symbols


def make_staging_dir(split: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f".{split}-", dir=OUTPUT_DIR))


def process_split(
    split: str,
    db: sqlite3.Connection,
    extractor: PEFeatureExtractor,
) -> dict[str, Any]:
    paths = gather_feature_paths(RAW_DIR, split)
    path_list = [Path(path) for path in paths]
    expected_rows, dimension = split_memmap_shape(split)
    raw_files = [{"path": path.relative_to(RAW_DIR).as_posix(), "size_bytes": path.stat().st_size} for path in path_list]
    x_memmap = np.memmap(RAW_DIR / f"X_{split}.dat", dtype=np.float32, mode="r", shape=(expected_rows, dimension))
    y_memmap = np.memmap(RAW_DIR / f"y_{split}.dat", dtype=np.int32, mode="r", shape=(expected_rows,))

    stage_dir = make_staging_dir(split)
    temporary_paths = {
        "metadata": stage_dir / "input_metadata.parquet",
        "vectors": stage_dir / "input_research_vectors.parquet",
        "imports": stage_dir / "input_imports.parquet",
    }
    final_paths = {
        "metadata": OUTPUT_DIR / f"input_metadata_{split}.parquet",
        "vectors": OUTPUT_DIR / f"input_research_vectors_{split}.parquet",
        "imports": OUTPUT_DIR / f"input_imports_{split}.parquet",
    }

    meta_columns = {field.name: [] for field in METADATA_SCHEMA}
    vector_rows: list[tuple[int, str, int, np.ndarray]] = []
    import_rows: list[tuple[int, str, str]] = []
    selected_locations: list[dict[str, Any]] = []
    raw_rows = 0
    unique_rows = 0
    duplicate_rows = 0
    label_disagreement_rows = 0
    selected_rows = 0
    selected_rows_without_imports = 0
    import_count = 0
    vector_nonfinite_values = 0
    vector_label_mismatches = 0
    metadata_rows_written = 0
    vector_rows_written = 0
    import_rows_written = 0
    started = time.monotonic()
    last_progress_rows = 0

    db.execute("BEGIN")
    try:
        metadata_writer = pq.ParquetWriter(temporary_paths["metadata"], METADATA_SCHEMA, compression="snappy")
        vector_writer = pq.ParquetWriter(temporary_paths["vectors"], VECTOR_SCHEMA, compression="snappy")
        import_writer = pq.ParquetWriter(temporary_paths["imports"], IMPORT_SCHEMA, compression="snappy")
        try:
            for file_index, path in enumerate(path_list):
                with path.open("rb") as source:
                    while True:
                        byte_offset = source.tell()
                        line = source.readline()
                        if not line:
                            break
                        source_id = raw_rows
                        raw_rows += 1
                        if not line.strip():
                            raise ValueError(f"blank JSONL row at {path}:{source_id}")
                        record = orjson.loads(line)
                        digest_hex, digest = raw_sha256(record, split, source_id)
                        label = int(record["label"])
                        if raw_rows % SQL_COMMIT_ROWS == 0:
                            db.commit()
                            db.execute("BEGIN")
                        if raw_rows - last_progress_rows >= 500_000:
                            last_progress_rows = raw_rows
                            elapsed = time.monotonic() - started
                            print(
                                f"{split}: {raw_rows:,} raw rows, {unique_rows:,} unique, "
                                f"{selected_rows:,} selected in {elapsed:.1f}s",
                                flush=True,
                            )
                        cursor = db.execute(
                            "INSERT OR IGNORE INTO unique_hash(split, digest, label) VALUES (?, ?, ?)",
                            (split, digest, label),
                        )
                        if cursor.rowcount == 0:
                            duplicate_rows += 1
                            retained = db.execute(
                                "SELECT label FROM unique_hash WHERE split = ? AND digest = ?", (split, digest)
                            ).fetchone()[0]
                            if label != retained:
                                label_disagreement_rows += 1
                                db.execute(
                                    "INSERT OR IGNORE INTO label_conflict(split, digest) VALUES (?, ?)",
                                    (split, digest),
                                )
                            continue

                        unique_rows += 1
                        source_fields = (
                            record.get("file_type"),
                            record.get("week_id"),
                            record.get("family"),
                            record.get("first_submission_date"),
                        )
                        if source_fields[0] is None or source_fields[1] is None or source_fields[3] is None:
                            raise ValueError(f"missing required metadata field at {split} source_id {source_id}")
                        detected, total_engines = parse_detection_ratio(record.get("detection_ratio"), split, source_id)
                        row_id = ID_BASE[split] + source_id
                        values = {
                            "id": row_id,
                            "source_id": source_id,
                            "split": split,
                            "sha256": digest_hex,
                            "file_type": str(source_fields[0]),
                            "week_id": int(source_fields[1]),
                            "label": label,
                            "family": "" if source_fields[2] is None else str(source_fields[2]),
                            "family_confidence": float(record.get("family_confidence") or 0.0),
                            "detected": detected,
                            "total_engines": total_engines,
                            "first_submission_date": int(source_fields[3]),
                        }
                        for name, value in values.items():
                            meta_columns[name].append(value)
                        if len(meta_columns["id"]) >= METADATA_CHUNK_ROWS:
                            metadata_rows_written += write_metadata_chunk(metadata_writer, meta_columns)

                        if is_selected(split, digest_hex):
                            selected_index = selected_rows
                            selected_rows += 1
                            selected_locations.append(
                                {
                                    "selection_index": selected_index,
                                    "source_id": source_id,
                                    "id": row_id,
                                    "file_index": file_index,
                                    "byte_offset": byte_offset,
                                    "sha256": digest_hex,
                                    "label": label,
                                }
                            )
                            mapped_label = int(y_memmap[source_id])
                            if mapped_label != label:
                                vector_label_mismatches += 1
                                raise ValueError(
                                    f"raw and y memmap labels differ at {split} source_id {source_id}: "
                                    f"{label} != {mapped_label}"
                                )
                            vector = np.asarray(x_memmap[source_id], dtype=np.float32)
                            if vector.shape != (VECTOR_DIM,):
                                raise ValueError(f"invalid vector shape at {split} source_id {source_id}: {vector.shape}")
                            vector_nonfinite_values += int((~np.isfinite(vector)).sum())
                            if vector_nonfinite_values:
                                raise ValueError(f"non-finite vector value at {split} source_id {source_id}")
                            vector_rows.append((row_id, split, label, vector.copy()))
                            if len(vector_rows) >= VECTOR_CHUNK_ROWS:
                                vector_rows_written += write_vector_chunk(vector_writer, vector_rows)

                            if split == "challenge":
                                symbols = extract_import_symbols(record.get("imports"))
                                if not symbols:
                                    selected_rows_without_imports += 1
                                for symbol in symbols:
                                    import_rows.append((row_id, split, symbol))
                                import_count += len(symbols)
                                if len(import_rows) >= IMPORT_CHUNK_ROWS:
                                    import_rows_written += write_import_chunk(import_writer, import_rows)

                if (file_index + 1) % 20 == 0 or file_index + 1 == len(path_list):
                    elapsed = time.monotonic() - started
                    print(
                        f"{split}: processed {file_index + 1}/{len(path_list)} files, {raw_rows:,} raw rows, "
                        f"{unique_rows:,} unique in {elapsed:.1f}s",
                        flush=True,
                    )

            if raw_rows != expected_rows:
                raise ValueError(f"{split} JSONL rows ({raw_rows}) do not match memmap rows ({expected_rows})")
            metadata_rows_written += write_metadata_chunk(metadata_writer, meta_columns)
            vector_rows_written += write_vector_chunk(vector_writer, vector_rows)
            import_rows_written += write_import_chunk(import_writer, import_rows)
        finally:
            metadata_writer.close()
            vector_writer.close()
            import_writer.close()
        db.commit()
    except Exception:
        db.rollback()
        raise

    if metadata_rows_written != unique_rows:
        raise ValueError(f"{split} metadata output has {metadata_rows_written} rows, expected {unique_rows}")
    if vector_rows_written != selected_rows:
        raise ValueError(f"{split} vector output has {vector_rows_written} rows, expected {selected_rows}")
    if import_rows_written != import_count:
        raise ValueError(f"{split} import output has {import_rows_written} rows, expected {import_count}")

    conflict_hashes = db.execute("SELECT COUNT(*) FROM label_conflict WHERE split = ?", (split,)).fetchone()[0]
    evidence = validate_sample_rows(split, selected_locations, path_list, x_memmap, y_memmap, extractor)
    output_counts: dict[str, int] = {}
    output_sizes: dict[str, int] = {}
    for kind, staged_path in temporary_paths.items():
        parquet_rows = pq.ParquetFile(staged_path).metadata.num_rows
        expected_count = {
            "metadata": unique_rows,
            "vectors": selected_rows,
            "imports": import_count,
        }[kind]
        if parquet_rows != expected_count:
            raise ValueError(f"{split} {kind} parquet row count {parquet_rows} != {expected_count}")
        final_paths[kind].parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_path, final_paths[kind])
        output_counts[kind] = parquet_rows
        output_sizes[kind] = final_paths[kind].stat().st_size
    stage_dir.rmdir()

    print(
        f"{split}: complete, raw={raw_rows:,}, unique={unique_rows:,}, selected={selected_rows:,}, "
        f"imports={import_count:,}, duplicate_label_disagreements={label_disagreement_rows:,}",
        flush=True,
    )
    del x_memmap, y_memmap
    return {
        "raw_row_count": raw_rows,
        "unique_row_count": unique_rows,
        "duplicate_row_count": duplicate_rows,
        "duplicate_label_disagreement_occurrences": label_disagreement_rows,
        "duplicate_label_disagreement_hashes": int(conflict_hashes),
        "selected_vector_row_count": selected_rows,
        "selected_rows_without_imports": selected_rows_without_imports,
        "import_symbol_row_count": import_count,
        "raw_files": raw_files,
        "memmaps": {
            "X": {"path": f"raw/X_{split}.dat", "size_bytes": (RAW_DIR / f"X_{split}.dat").stat().st_size},
            "y": {"path": f"raw/y_{split}.dat", "size_bytes": (RAW_DIR / f"y_{split}.dat").stat().st_size},
            "rows": expected_rows,
            "feature_dimension": dimension,
        },
        "outputs": {
            kind: {
                "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
                "rows": output_counts[kind],
                "size_bytes": output_sizes[kind],
            }
            for kind, path in final_paths.items()
        },
        "validation": {
            "jsonl_rows_match_memmaps": raw_rows == expected_rows,
            "metadata_rows_match_unique_hashes": metadata_rows_written == unique_rows,
            "selected_labels_match_y_memmap": vector_label_mismatches == 0,
            "vector_nonfinite_value_count": vector_nonfinite_values,
            "raw_feature_evidence": evidence,
        },
    }


def validate_sample_rows(
    split: str,
    selected_locations: list[dict[str, Any]],
    paths: list[Path],
    x_memmap: np.memmap,
    y_memmap: np.memmap,
    extractor: PEFeatureExtractor,
) -> list[dict[str, Any]]:
    if not selected_locations:
        raise ValueError(f"{split} has no selected records to validate")
    last = len(selected_locations) - 1
    sample_indexes = {"first": 0, "middle": len(selected_locations) // 2, "last": last}
    results: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    for position, index in sample_indexes.items():
        if index in seen_indexes:
            continue
        seen_indexes.add(index)
        location = selected_locations[index]
        source_id = location["source_id"]
        with paths[location["file_index"]].open("rb") as source:
            source.seek(location["byte_offset"])
            raw_line = source.readline()
        raw_record = orjson.loads(raw_line)
        raw_digest, _ = raw_sha256(raw_record, split, source_id)
        raw_label = int(raw_record["label"])
        mapped_label = int(y_memmap[source_id])
        if raw_digest != location["sha256"] or raw_label != location["label"] or raw_label != mapped_label:
            raise ValueError(f"{split} {position} sample lookup did not return its selected raw row")
        raw_vector = np.asarray(extractor.process_raw_features(raw_record), dtype=np.float32)
        memmap_vector = np.asarray(x_memmap[source_id], dtype=np.float32)
        if raw_vector.shape != (VECTOR_DIM,):
            raise ValueError(f"raw feature extractor returned {raw_vector.shape} at {split} source_id {source_id}")
        difference = np.abs(raw_vector - memmap_vector)
        max_abs_difference = float(difference.max(initial=0.0))
        if not np.allclose(raw_vector, memmap_vector, rtol=1e-6, atol=1e-7, equal_nan=False):
            raise ValueError(
                f"{split} {position} raw feature evidence differs from memmap at source_id {source_id}; "
                f"max_abs_difference={max_abs_difference}"
            )
        results.append(
            {
                "position": position,
                "selection_index": index,
                "id": location["id"],
                "source_id": source_id,
                "sha256": raw_digest,
                "raw_label": raw_label,
                "memmap_label": mapped_label,
                "feature_dimension": int(raw_vector.size),
                "raw_vs_memmap_exact_equal": bool(np.array_equal(raw_vector, memmap_vector)),
                "raw_vs_memmap_allclose": True,
                "max_absolute_feature_difference": max_abs_difference,
            }
        )
    return results


def validate_weekly_file_first_rows(extractor: PEFeatureExtractor) -> dict[str, Any]:
    """Check file boundaries and selected raw evidence against their memmap ordinals."""
    checks_by_split: dict[str, Any] = {}
    for split in SPLITS:
        paths = [Path(path) for path in gather_feature_paths(RAW_DIR, split)]
        expected_rows, dimension = split_memmap_shape(split)
        x_memmap = np.memmap(
            RAW_DIR / f"X_{split}.dat", dtype=np.float32, mode="r", shape=(expected_rows, dimension)
        )
        y_memmap = np.memmap(RAW_DIR / f"y_{split}.dat", dtype=np.int32, mode="r", shape=(expected_rows,))

        vector_path = OUTPUT_DIR / f"input_research_vectors_{split}.parquet"
        selected = pq.read_table(vector_path, columns=["id", "label"])
        selected_ids = selected.column("id").to_numpy()
        selected_labels = selected.column("label").to_numpy()
        if not len(selected_ids):
            raise ValueError(f"{split} has no selected vector rows")
        sample_indexes = {"first": 0, "middle": len(selected_ids) // 2, "last": len(selected_ids) - 1}
        sample_targets: dict[int, dict[str, Any]] = {}
        for position, sample_index in sample_indexes.items():
            row_id = int(selected_ids[sample_index])
            source_sample_id = row_id - ID_BASE[split]
            target = sample_targets.setdefault(
                source_sample_id,
                {"selection_index": sample_index, "id": row_id, "label": int(selected_labels[sample_index]), "positions": []},
            )
            target["positions"].append(position)

        source_id = 0
        file_checks = []
        file_ranges: list[dict[str, Any]] = []
        started = time.monotonic()
        for file_index, path in enumerate(paths):
            with path.open("rb") as source:
                first_line = source.readline()
                if not first_line:
                    raise ValueError(f"empty JSONL file: {path}")
                row_count = first_line.count(b"\n")
                last_byte = first_line[-1:]
                while True:
                    block = source.read(16 * 1024 * 1024)
                    if not block:
                        break
                    row_count += block.count(b"\n")
                    last_byte = block[-1:]
                if last_byte != b"\n":
                    row_count += 1
            if row_count == 0:
                raise ValueError(f"empty JSONL file: {path}")
            record = orjson.loads(first_line)
            digest_hex, _ = raw_sha256(record, split, source_id)
            raw_label = int(record["label"])
            mapped_label = int(y_memmap[source_id])
            if raw_label != mapped_label:
                raise ValueError(f"raw and y memmap labels differ at {split} source_id {source_id}")
            raw_vector = np.asarray(extractor.process_raw_features(record), dtype=np.float32)
            memmap_vector = np.asarray(x_memmap[source_id], dtype=np.float32)
            if raw_vector.shape != (VECTOR_DIM,):
                raise ValueError(f"raw extractor returned {raw_vector.shape} at {split} source_id {source_id}")
            difference = np.abs(raw_vector - memmap_vector)
            max_abs_difference = float(difference.max(initial=0.0))
            if not np.allclose(raw_vector, memmap_vector, rtol=1e-6, atol=1e-7, equal_nan=False):
                raise ValueError(
                    f"first raw row in {path.name} differs from memmap at source_id {source_id}; "
                    f"max_abs_difference={max_abs_difference}"
                )
            file_checks.append(
                {
                    "path": path.relative_to(RAW_DIR).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "file_raw_row_count": row_count,
                    "source_id": source_id,
                    "sha256": digest_hex,
                    "raw_label": raw_label,
                    "memmap_label": mapped_label,
                    "raw_vs_memmap_exact_equal": bool(np.array_equal(raw_vector, memmap_vector)),
                    "max_absolute_feature_difference": max_abs_difference,
                }
            )
            file_ranges.append({"path": path, "start": source_id, "end": source_id + row_count})
            source_id += row_count
            if (file_index + 1) % 20 == 0 or file_index + 1 == len(paths):
                print(
                    f"{split} first-row evidence: checked {file_index + 1}/{len(paths)} files "
                    f"in {time.monotonic() - started:.1f}s",
                    flush=True,
                )
        if source_id != expected_rows:
            raise ValueError(f"{split} JSONL file row counts sum to {source_id}, expected {expected_rows}")
        sample_evidence: dict[str, dict[str, Any]] = {}
        for source_sample_id, target in sample_targets.items():
            file_range = next(
                (entry for entry in file_ranges if entry["start"] <= source_sample_id < entry["end"]), None
            )
            if file_range is None:
                raise ValueError(f"selected source_id {source_sample_id} does not map to a raw file in {split}")
            local_source_id = source_sample_id - file_range["start"]
            with file_range["path"].open("rb") as source:
                raw_line = b""
                for _ in range(local_source_id + 1):
                    raw_line = source.readline()
                if not raw_line:
                    raise ValueError(f"could not seek to {split} source_id {source_sample_id}")
            record = orjson.loads(raw_line)
            digest_hex, _ = raw_sha256(record, split, source_sample_id)
            raw_label = int(record["label"])
            mapped_label = int(y_memmap[source_sample_id])
            if raw_label != target["label"] or raw_label != mapped_label:
                raise ValueError(f"raw, vector parquet, and y memmap labels differ at {split} source_id {source_sample_id}")
            if not is_selected(split, digest_hex):
                raise ValueError(f"selected raw sample violates the selection rule at {split} source_id {source_sample_id}")
            raw_vector = np.asarray(extractor.process_raw_features(record), dtype=np.float32)
            memmap_vector = np.asarray(x_memmap[source_sample_id], dtype=np.float32)
            if raw_vector.shape != (VECTOR_DIM,):
                raise ValueError(f"raw extractor returned {raw_vector.shape} at {split} source_id {source_sample_id}")
            difference = np.abs(raw_vector - memmap_vector)
            max_abs_difference = float(difference.max(initial=0.0))
            if not np.allclose(raw_vector, memmap_vector, rtol=1e-6, atol=1e-7, equal_nan=False):
                raise ValueError(
                    f"selected raw feature evidence differs from memmap at {split} source_id {source_sample_id}; "
                    f"max_abs_difference={max_abs_difference}"
                )
            for position in target["positions"]:
                sample_evidence[position] = {
                    "position": position,
                    "selection_index": target["selection_index"],
                    "id": target["id"],
                    "source_id": source_sample_id,
                    "sha256": digest_hex,
                    "raw_label": raw_label,
                    "memmap_label": mapped_label,
                    "feature_dimension": int(raw_vector.size),
                    "raw_vs_memmap_exact_equal": bool(np.array_equal(raw_vector, memmap_vector)),
                    "raw_vs_memmap_allclose": True,
                    "max_absolute_feature_difference": max_abs_difference,
                }
        checks_by_split[split] = {
            "file_count": len(paths),
            "raw_rows_match_memmap": source_id == expected_rows,
            "first_rows_match_memmap": len(file_checks) == len(paths),
            "files": file_checks,
            "selected_rows_match_memmap": len(sample_evidence) == len(set(sample_indexes)),
            "selected_raw_feature_evidence": [sample_evidence[position] for position in sample_indexes],
        }
        if len(sample_evidence) != len(set(sample_indexes)):
            raise ValueError(f"missing selected sample evidence for {split}")
        del x_memmap, y_memmap
    return checks_by_split


def combine_parquet_outputs(split_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for kind, stem in (
        ("metadata", "input_metadata"),
        ("vectors", "input_research_vectors"),
        ("imports", "input_imports"),
    ):
        final_path = OUTPUT_DIR / f"{stem}.parquet"
        temp_path = OUTPUT_DIR / f".{stem}.parquet.tmp"
        split_paths = [OUTPUT_DIR / f"{stem}_{split}.parquet" for split in SPLITS]
        schemas = [pq.ParquetFile(path).schema_arrow for path in split_paths]
        if any(schema != schemas[0] for schema in schemas[1:]):
            raise ValueError(f"incompatible {kind} schemas across split outputs")
        with pq.ParquetWriter(temp_path, schemas[0], compression="snappy") as writer:
            for split_path in split_paths:
                parquet = pq.ParquetFile(split_path)
                for batch in parquet.iter_batches(batch_size=50_000):
                    writer.write_batch(batch)
        os.replace(temp_path, final_path)
        rows = pq.ParquetFile(final_path).metadata.num_rows
        expected_rows = sum(split_results[split]["outputs"][kind]["rows"] for split in SPLITS)
        if rows != expected_rows:
            raise ValueError(f"combined {kind} parquet has {rows} rows, expected {expected_rows}")
        combined[kind] = {
            "path": final_path.relative_to(WORKSPACE_ROOT).as_posix(),
            "rows": rows,
            "size_bytes": final_path.stat().st_size,
        }
    return combined


def exported_fixture_parquet_counts() -> dict[str, Any]:
    parquet_dir = FIXTURE_ROOT / "parquet"
    files = []
    if parquet_dir.is_dir():
        for path in sorted(parquet_dir.rglob("*.parquet"), key=lambda item: item.relative_to(parquet_dir).as_posix()):
            files.append(
                {
                    "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
                    "rows": pq.ParquetFile(path).metadata.num_rows,
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "directory_present": parquet_dir.is_dir(),
        "files": files,
        "total_rows": sum(item["rows"] for item in files),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    extractor = PEFeatureExtractor()
    if extractor.dim != VECTOR_DIM:
        raise ValueError(f"thrember extractor has {extractor.dim} features, expected {VECTOR_DIM}")

    state_fd, state_name = tempfile.mkstemp(prefix="ember2024-research-", suffix=".sqlite", dir=OUTPUT_DIR)
    os.close(state_fd)
    db = sqlite3.connect(state_name)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=MEMORY")
    db.execute(
        "CREATE TABLE unique_hash (split TEXT NOT NULL, digest BLOB NOT NULL, label INTEGER NOT NULL, "
        "PRIMARY KEY(split, digest)) WITHOUT ROWID"
    )
    db.execute(
        "CREATE TABLE label_conflict (split TEXT NOT NULL, digest BLOB NOT NULL, "
        "PRIMARY KEY(split, digest)) WITHOUT ROWID"
    )
    split_results: dict[str, dict[str, Any]] = {}
    try:
        print(
            f"EMBER2024 research preparation: {RAW_DIR}, outputs {OUTPUT_DIR}; "
            f"orjson={package_version('orjson')}, pyarrow={package_version('pyarrow')}",
            flush=True,
        )
        for split in SPLITS:
            split_results[split] = process_split(split, db, extractor)

        per_file_first_rows = validate_weekly_file_first_rows(extractor)
        overlap_pairs = []
        pairwise_overlap_total = 0
        for left_split, right_split in (("challenge", "test"), ("challenge", "train"), ("test", "train")):
            shared_hashes = db.execute(
                "SELECT COUNT(*) FROM unique_hash AS left_side "
                "JOIN unique_hash AS right_side ON left_side.digest = right_side.digest "
                "WHERE left_side.split = ? AND right_side.split = ?",
                (left_split, right_split),
            ).fetchone()[0]
            pairwise_overlap_total += shared_hashes
            overlap_pairs.append((left_split, right_split, shared_hashes))
        triple_overlap = db.execute(
            "SELECT COUNT(*) FROM unique_hash AS challenge "
            "JOIN unique_hash AS test ON test.split = 'test' AND test.digest = challenge.digest "
            "JOIN unique_hash AS train ON train.split = 'train' AND train.digest = challenge.digest "
            "WHERE challenge.split = 'challenge'"
        ).fetchone()[0]
        cross_split_distinct_hashes = pairwise_overlap_total - 2 * triple_overlap

        combined_outputs = combine_parquet_outputs(split_results)
        id_range_checks = {}
        for split in SPLITS:
            raw_count = split_results[split]["raw_row_count"]
            if raw_count >= 10_000_000:
                raise ValueError(f"{split} ID range would overlap the next split's reserved base")
            id_range_checks[split] = {
                "base": ID_BASE[split],
                "raw_rows_below_next_base": raw_count,
                "range_disjoint": True,
            }

        provenance = {
            "dataset": "EMBER2024",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "workspace_root": str(WORKSPACE_ROOT),
            "raw_directory": RAW_DIR.relative_to(WORKSPACE_ROOT).as_posix(),
            "processing_order": list(SPLITS),
            "path_order": "thrember.model.gather_feature_paths, which sorts raw JSONL names lexically",
            "packages": {
                "python": sys.version.split()[0],
                "thrember": package_version("thrember"),
                "thrember_vcs_commit": vcs_commit("thrember"),
                "orjson": package_version("orjson"),
                "numpy": package_version("numpy"),
                "pyarrow": package_version("pyarrow"),
                "networkx": package_version("networkx"),
            },
            "selection_rule": {
                "challenge": "all unique SHA256 rows",
                "train_test": "all unique SHA256 rows",
                "selection_uses_label_or_family": False,
                "deduplication": "first row for each SHA256 within each split is retained",
            },
            "id_rule": {
                "source_id": "raw row ordinal within the split, incremented once for every JSONL row across sorted files",
                "id_bases": ID_BASE,
                "global_ids_unique": all(check["range_disjoint"] for check in id_range_checks.values()),
                "range_checks": id_range_checks,
            },
            "counts": {
                "splits": split_results,
                "total_raw_rows": sum(item["raw_row_count"] for item in split_results.values()),
                "total_unique_rows": sum(item["unique_row_count"] for item in split_results.values()),
                "cross_split_distinct_sha256_hashes": int(cross_split_distinct_hashes),
                "cross_split_hash_overlaps": [
                    {"left_split": row[0], "right_split": row[1], "shared_sha256_hashes": row[2]}
                    for row in overlap_pairs
                ],
            },
            "exports": {
                "combined_outputs": combined_outputs,
                "exported_fixture_parquet_row_counts": exported_fixture_parquet_counts(),
            },
            "schema_contracts": {
                "metadata_fields_nonnull": [field.name for field in METADATA_SCHEMA],
                "vector_field": {"name": "vector", "type": f"fixed_size_list<float32>[{VECTOR_DIM}]", "nullable": False},
                "histogram_field": {
                    "name": "histogram",
                    "type": f"fixed_size_list<float32>[{HISTOGRAM_DIM}]",
                    "source_dimensions_zero_based": [HISTOGRAM_START, HISTOGRAM_STOP - 1],
                    "nullable": False,
                },
                "imports": {
                    "fields": ["id", "split", "symbol"],
                    "symbol_rule": "lowercased DLL name, exclamation mark, then API name as supplied",
                    "duplicate_symbols": "repeated symbols are removed within each selected sample",
                    "ordinal_api_names": "retained as supplied when present; they are not filtered",
                },
            },
            "validation": {
                "all_split_validations_passed": all(
                    result["validation"]["jsonl_rows_match_memmaps"]
                    and result["validation"]["metadata_rows_match_unique_hashes"]
                    and result["validation"]["selected_labels_match_y_memmap"]
                    and result["validation"]["vector_nonfinite_value_count"] == 0
                    and len(result["validation"]["raw_feature_evidence"]) == 3
                    for result in split_results.values()
                )
                and all(
                    result["raw_rows_match_memmap"]
                    and result["first_rows_match_memmap"]
                    and result["selected_rows_match_memmap"]
                    for result in per_file_first_rows.values()
                ),
                "per_raw_file_first_row_checks_passed": all(
                    result["raw_rows_match_memmap"] and result["first_rows_match_memmap"]
                    for result in per_file_first_rows.values()
                ),
                "per_raw_file_first_rows": per_file_first_rows,
                "global_id_range_checks_passed": all(check["range_disjoint"] for check in id_range_checks.values()),
                "raw_feature_evidence_comparison": "first, middle, and last selected row per split compared with PEFeatureExtractor.process_raw_features; rtol=1e-6, atol=1e-7",
                "per_split": {split: result["validation"] for split, result in split_results.items()},
            },
        }
        provenance_tmp = PROVENANCE_PATH.with_suffix(".json.tmp")
        provenance_tmp.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf8")
        os.replace(provenance_tmp, PROVENANCE_PATH)
        print(f"provenance: {PROVENANCE_PATH}", flush=True)
        for kind, result in combined_outputs.items():
            print(f"{kind}: {result['rows']:,} rows, {result['size_bytes']:,} bytes", flush=True)
        print(f"exported fixture Parquet rows: {provenance['exports']['exported_fixture_parquet_row_counts']['total_rows']:,}", flush=True)
    finally:
        db.close()
        Path(state_name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
