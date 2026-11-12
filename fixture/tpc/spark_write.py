#!/usr/bin/env python3
"""Write one table as a single Parquet file with parquet-java through PySpark.

`parquet_layout.write_table_with_spark` runs this script under the interpreter
that has PySpark and passes a JSON job file. The output rows keep the order of
the input files and, within each file, the order of its rows.
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text())
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("algeon-tpc-fixture")
        .config("spark.driver.memory", job["driver_memory"])
        .config("spark.local.dir", job["local_dir"])
        .config("spark.ui.enabled", "false")
        # Spark sorts file splits by size, largest first, before packing them into
        # read partitions, so one read over several files or one file cut into
        # unequal splits can reorder rows. These two settings give each file
        # exactly one split, and the per-file reads below are unioned in order.
        .config("spark.sql.files.maxPartitionBytes", str(2**62))
        .config("spark.sql.files.minPartitionNum", "1")
        .config("spark.sql.parquet.compression.codec", job["compression"])
        .config("spark.sql.parquet.writeLegacyFormat", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        frames = []
        for path in job["inputs"]:
            frame = spark.read.parquet(path)
            partitions = frame.rdd.getNumPartitions()
            if partitions != 1:
                raise RuntimeError(f"Spark planned {partitions} read partitions for {path}; row order needs exactly one")
            frames.append(frame)
        rows = functools.reduce(lambda left, right: left.union(right), frames)
        # coalesce(1) without a shuffle concatenates the union's partitions in order
        # because local files carry no preferred locations.
        (
            rows.coalesce(1)
            .write.mode("errorifexists")
            .option("parquet.block.size", str(job["parquet_block_size"]))
            .option("parquet.page.size", str(job["parquet_page_size"]))
            .option("parquet.page.row.count.limit", str(job["parquet_page_row_count_limit"]))
            .parquet(job["output_dir"])
        )
        Path(job["result_file"]).write_text(json.dumps({"spark_version": spark.version}) + "\n")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
