"""Generate lineage.dot (dataset-level lineage) and render docs/lineage.png.

    python docs/diagrams/make_lineage.py      # needs Graphviz `dot` on PATH
Row counts are those of the reference run (identical to the original notebook).
"""
from __future__ import annotations

import html
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

DATA = {"fill": "#f4f6f9", "border": "#5b6b7f"}
PRODUCT = {"fill": "#e9f7e9", "border": "#3d8b3d"}
SOURCE = {"fill": "#fff4e0", "border": "#c98a1b"}
DB = {"fill": "#e8f0fe", "border": "#2f6db5"}


def table(title: str, subtitle: str, rows: list[str], style: dict) -> str:
    cells = "".join(f'<TR><TD ALIGN="LEFT" BALIGN="LEFT">{html.escape(r)}</TD></TR>' for r in rows)
    return (f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" COLOR="{style["border"]}" '
            f'BGCOLOR="{style["fill"]}"><TR><TD ALIGN="LEFT"><B>{html.escape(title)}</B></TD></TR>'
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="#444444">{html.escape(subtitle)}</FONT></TD></TR>'
            f'<HR/>{cells}</TABLE>>')


def step(title: str, lines: list[str], color: str) -> str:
    body = "<BR ALIGN=\"LEFT\"/>".join(html.escape(l) for l in lines)
    return (f'<<TABLE BORDER="0" CELLPADDING="5" BGCOLOR="{color}"><TR><TD ALIGN="LEFT">'
            f'<B>&#9654; {html.escape(title)}</B><BR ALIGN="LEFT"/><FONT POINT-SIZE="9">{body}</FONT>'
            f'<BR ALIGN="LEFT"/></TD></TR></TABLE>>')


NODES = {
    "src": table("SOURCE  world_air_quality.csv", "54,255 rows · sha256 2b2202a0… · checksum-verified",
                 ["Country Code · City · Location · Coordinates", "Pollutant · Source Name · Unit · Value",
                  "Last Updated · Country Label"], SOURCE),
    "t_ing": step("ingest_source", ["verbatim strings + lineage columns", "full | incremental (source-row high-water mark)"], "#fdebd0"),
    "raw": table("RAW  data/raw/committed/b*.parquet", "54,255 rows · key (_source_sha256, _source_row_number)",
                 ["10 source fields as exact strings", "_source_row_number · _record_hash",
                  "_ingest_batch_id · _pipeline_run_id · _ingested_at"], DATA),
    "t_vr": step("validate_raw", ["9 data-contract checks", "pass → commit batch + advance checkpoint", "fail → quarantine"], "#fdebd0"),
    "t_stg": step("transform_staging  (notebook rules, notebook order)", [
        "1  implicit NA ('N/A', 'None' → NULL)", "2  pollutant in {PM2.5, PM10, NO2, O3, SO2, CO}   → 47,870",
        "3  Last Updated → UTC timestamp", "4  de-dup (Location, Pollutant, ts), keep first   → 39,796",
        "5  unit = µg/m³   → 30,853", "6  value ≥ 0   → 30,545", "7  City / Location / Country NULL → 'Unknown'"], "#e3f1e3"),
    "stg": table("STAGING  air_quality_readings", "30,545 rows · key (location, pollutant, last_updated_utc)",
                 ["reading_key · country_code · country_label · city · location",
                  "latitude, longitude  (from Coordinates; not a model input)",
                  "pollutant · source_name · unit · value", "last_updated_utc → reading_date"], DATA),
    "t_cur": step("build_curated", ["group by station-day, daily mean", "pivot pollutant → columns",
                                    "sort (location, date) = notebook row order"], "#e3f1e3"),
    "cur": table("CURATED  station_day_pollutants  (data product)", "11,558 rows · key (country_label, city, location, reading_date)",
                 ["station_day_key", "co · no2 · o3 · pm10 · pm25 · so2  (daily means; NULL if not reported)", "n_readings"], PRODUCT),
    "t_mf": step("model features", ["require pm25, no2, o3 present", "median-impute SO2, CO + _missing flags",
                                    "Ox_total_oxidant = NO2 + O3 ;  NO2_O3_ratio = NO2 / (O3 + 1)",
                                    "label: PM2.5 > 15 µg/m³ → Unhealthy, else Safe"], "#e3f1e3"),
    "mf": table("MODEL_FEATURES", "2,628 rows · 1,513 Unhealthy / 1,115 Safe",
                ["X: NO2, O3, SO2, CO, SO2_missing, CO_missing,", "    Ox_total_oxidant, NO2_O3_ratio  (contract order)",
                 "y: Air_Quality  (derived from PM2.5)", "excluded: PM10 — leakage (physically contains PM2.5)"], DATA),
    "t_split": step("train_model", ["stratified 80/20 split, seed 42", "train/test overlap + feature-dimension checks"], "#eaf2fb"),
    "split": table("SPLIT  split.csv", "train 2,102 · test 526", ["test-key fingerprint 9b330835… = original notebook"], DATA),
    "models": table("MODELS  models/*.joblib", "fitted on the 2,102 training rows",
                    ["Logistic Regression (max_iter 1000)", "Random Forest (200 trees, seed 42)",
                     "Gradient Boosting (200 stages, seed 42)"], DATA),
    "t_eval": step("evaluate_model", ["held-out metrics on 526 test rows", "5-fold stratified CV (F1-weighted)"], "#eaf2fb"),
    "preds": table("PREDICTIONS", "test rows × 3 models", ["predictions.csv (526 rows)",
                                                          "ml.predictions (1,578 rows; key station_day_key + model)"], PRODUCT),
    "metrics": table("METRICS", "per model", ["metrics.json · ml.model_metrics"], PRODUCT),
    "base": table("BASELINE  baseline_metrics.json", "from the archived notebook",
                  ["saved outputs (paper Tables I–II)", "+ verbatim re-execution (split fingerprint)"], SOURCE),
    "t_rep": step("reproducibility_check", ["counts + split membership exact", "|Δ metric| ≤ 0.01"], "#eaf2fb"),
    "comp": table("metrics_comparison.json", "status: pass", ["15 metrics, 14 exact, max |Δ| = 0.0004"], PRODUCT),
    "pg_stg": table("PostgreSQL", "staging.air_quality_readings", ["upsert on reading_key"], DB),
    "pg_cur": table("PostgreSQL", "curated.station_day_pollutants", ["upsert on station_day_key"], DB),
    "parts": table("PARTITIONS", "curated/partitions/country_label=<slug>/", ["109 partitions, readable independently"], DATA),
}
EDGES = [
    ("src", "t_ing", ""), ("t_ing", "raw", ""), ("raw", "t_vr", ""), ("t_vr", "t_stg", ""), ("t_stg", "stg", ""),
    ("stg", "t_cur", ""), ("t_cur", "cur", ""), ("cur", "t_mf", ""), ("t_mf", "mf", ""), ("mf", "t_split", ""),
    ("t_split", "split", ""), ("t_split", "models", ""), ("split", "t_eval", ""), ("models", "t_eval", ""),
    ("t_eval", "preds", ""), ("t_eval", "metrics", ""), ("metrics", "t_rep", ""), ("base", "t_rep", ""),
    ("t_rep", "comp", ""),
]
SIDE = [
    ('stg', 'pg_stg', 'label="load_postgres", color="#2f6db5"'),
    ('cur', 'pg_cur', 'label="load_postgres", color="#2f6db5"'),
    ('pg_cur', 't_mf', 'label="parity check (validate_curated)", style=dashed, color="#2f6db5"'),
    ('cur', 'parts', 'label="partition demo", style=dashed'),
]


def main() -> None:
    lines = [
        "// GENERATED by make_lineage.py — edit the script, not this file.",
        "digraph lineage {",
        '  graph [rankdir=TB, fontname="Helvetica", pad=0.3, nodesep=0.45, ranksep=0.3, bgcolor="white",',
        '         label="Data lineage — world_air_quality.csv to PM2.5 exceedance predictions (row counts from the reference run)",',
        '         labelloc=t, fontsize=18];',
        '  node [shape=plain, fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9, color="#5b6b7f"];',
    ]
    lines += [f"  {k} [label={v}];" for k, v in NODES.items()]
    lines += [f"  {a} -> {b};" for a, b, _ in EDGES]
    lines += [f"  {a} -> {b} [{attrs}];" for a, b, attrs in SIDE]
    lines += ["  {rank=same; stg; pg_stg;}", "  {rank=same; cur; pg_cur; parts;}", "}"]
    dot = HERE / "lineage.dot"
    dot.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(["dot", "-Tpng", "-Gdpi=130", str(dot), "-o", str(HERE.parent / "lineage.png")], check=True)
    print("wrote docs/diagrams/lineage.dot and docs/lineage.png")


if __name__ == "__main__":
    main()
