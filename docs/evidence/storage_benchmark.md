# Storage Experiment — CSV vs JSON vs Parquet vs PostgreSQL

**Command:** `python -m pipeline.cli benchmark --repeats 3` (after a full pipeline run)
**Code:** `src/pipeline/storage_benchmark.py` · raw results: `data/benchmark/storage_benchmark.json` / `.md`

## Method

* **Same logical records.** Two datasets produced by the pipeline are written to every target: `staging_readings` (30,545 rows, 13 business columns incl. a UTC timestamp and a date) and `curated_station_day` (11,558 rows, 12 columns with many NULLs). Audit columns are excluded.
* **Measured per format:** write time, full read time, on-disk size, and an analytical query (mean PM2.5 for Japan = read + filter + aggregate). Each timing is the median of 3 repeats.
* **Equivalence check (A10).** After each read-back the frame is coerced to the reference types and its **content fingerprint** (sha256 over business columns, sorted by key, floats to 10 significant digits) is compared with the in-memory original. The report also shows the **largest absolute float error**. A format "wins" only if it returns the same logical records.
* **PostgreSQL** is measured as `COPY` into an empty `bench.*` table (write), `SELECT *` into pandas (read), `pg_total_relation_size` (size incl. indexes/TOAST) and a SQL `avg()` (query).

## Results

### A. Docker environment — **authoritative** (fill in after your run)

Paste the table from `data/benchmark/storage_benchmark.md` here after running inside Docker (pyarrow installed, psycopg2 driver):

```
docker compose run --rm pipeline python -m pipeline.cli benchmark --repeats 3
```

| Dataset | Format | Rows | Write (s) | Read (s) | Query (s) | Size (KB) | Same records | Max float err | Status |
|---|---|---|---|---|---|---|---|---|---|
| _paste here_ | | | | | | | | | |

### B. Development sandbox (reference only)

Linux, Python 3.13, pandas 3.0.5. pyarrow could not be installed (no package index access), so the Parquet rows are skipped. PostgreSQL ran through a test-only psql-backed driver, so its read times include process overhead and are pessimistic.

| Dataset | Format | Rows | Write (s) | Read (s) | Query (s) | Size (KB) | Same records | Max float err |
|---|---|---|---|---|---|---|---|---|
| staging_readings | csv | 30,545 | 0.302 | 0.082 | 0.098 | 4,791 | True | 0 |
| staging_readings | json_lines | 30,545 | 0.332 | 0.325 | 0.255 | 11,064 | True | 8.9e-16 |
| staging_readings | postgresql | 30,545 | 0.434 | 0.314 | **0.005** | 5,192 | True | 0 |
| curated_station_day | csv | 11,558 | 0.085 | 0.027 | 0.027 | 1,150 | True | 2.2e-16 |
| curated_station_day | json_lines | 11,558 | 0.049 | 0.064 | 0.080 | 2,959 | True | 8.9e-16 |
| curated_station_day | postgresql | 11,558 | 0.114 | 0.116 | **0.002** | 1,736 | True | 2.2e-16 |

## Findings

1. **JSON is lossy by default.** The first benchmark run flagged JSON as *not* returning the same records. pandas' `to_json` defaults to `double_precision=10`, which rounds daily-mean concentrations such as 12.333333333333334. Only `double_precision=15` (the maximum) round-trips within 1e-15. JSON is also the largest format (about 2.3× CSV) and the slowest to read.
2. **PostgreSQL wins for selective queries.** Even through a slow driver, an indexed `avg()` over one country takes milliseconds because only the result crosses the wire. File formats must read and parse the whole file first. For full-table reads into pandas, files are faster.
3. **Parquet** (expected from the Docker run): smallest files thanks to columnar compression, and the fastest full reads. It is the only file format that keeps types (UTC timestamps, dates, NULLs) without any coercion code, which is why the pipeline uses it for its layers.
4. **CSV** is fast at this size and universally readable. But it needed explicit type restoration (dates, timestamps, empty string vs NULL) to be equivalent. That is the dtype sidecar in `layer_io.py`.

## Why this result should not be generalised

* **Tiny data.** About 5 MB fits in memory and the OS page cache, so timings mostly measure Python parsing overhead, not I/O. At 100× scale, columnar formats and predicate push-down dominate. Parquet would read only the needed columns and row groups, and PostgreSQL would need indexes and partitioning.
* **Single machine, warm cache, no concurrency.** A database's strengths (concurrent readers, transactions, constraints, incremental upserts) are invisible in a single-user benchmark.
* **Workload shape.** One write and many full reads favour files. Many small lookups or updates favour a database. Streaming appends favour row formats (CSV/JSON lines).
* **Environment.** Driver, compression codec, disk type, Docker-on-Windows bind-mount overhead and library versions all shift the numbers. Only the Docker run (§A) is representative of the delivered system.

## Partition demonstration (A11)

`python -m pipeline.cli partition --value Japan --load-db`

* Writes the curated table as Hive-style partitions `data/curated/partitions/country_label=<slug>/part-0.parquet` (109 partitions, with an `_index.json` mapping slug → label and row count).
* Reads **one** partition (1 file, 1,047 rows for Japan) and verifies its fingerprint equals the same slice filtered from the full table.
* Loads **only** that partition into PostgreSQL with a scope-limited merge (`load_frame(..., scope_column='country_label', scope_value='Japan')`). Deletes are restricted to that country, so other partitions are untouched. Sandbox result: 1,047 unchanged, 0 inserted/updated/deleted, table still 11,558 rows.
* Natural partition choice: `country_label` is the main analytical grouping. There is no meaningful time partition because the file is a snapshot: 82% of the 8,941 stations report on a single date (median 1).
