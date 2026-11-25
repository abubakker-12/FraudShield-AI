# Batch prediction capacity audit

## Approved public limits

The public CSV app accepts up to **5 MB (5,000,000 bytes)** and **25,000 records**, enforced independently. Streamlit's upload setting is 5, and the app validates the exact decimal byte limit. These are the approved public limits.

| Environment | Envelope | Evidence and margin |
|---|---:|---|
| Public Community Cloud | 5 MB and 25,000 records | Approved application limits. Container capacity and concurrency remain unverified. |
| Local benchmark, one worker | 25 MB and 100,000 records | Both limits exercised together with roughly 650 MiB available before starting. Benchmark only. |
| Local benchmark, one to four simulated sessions | 10 MB and 50,000 records per session | Historical local measurements only. |

Apply byte and record limits independently. A narrow CSV can exceed the record cap below the byte cap, while optional text can consume the byte cap early. The 250,000-row and 150 MB stress results are evidence of local processor capability under the measured conditions. They provide no endorsement for that combined size or for hosting those uploads.

## Existing implementation and bottlenecks

The initial release used a 2 MB upload setting and a separate 1,000-record app limit. There is no benchmark-derived 2 MB threshold in the repository history. It served as a conservative demo guard against memory and workload growth. The FastAPI `POST /predict-batch` limit remains 1,000 JSON feature objects and was outside the CSV benchmark.

The previous CSV path decoded the complete upload to a Unicode string, checked the CSV structure, parsed required columns with pandas, and computed features for validation. Scoring recomputed validation and features every 250 records. Prediction dictionaries were materialized per row, and the UI generated and passed all rows to chart, table and CSV export on reruns.

The new path streams strict CSV validation over the in-memory upload, retains only required columns, validates once and stores compact numeric and categorical dtypes. It engineers the same seven features at inference time: overnight, repeat failures, short interval, international/repeated failures, repeat failures/short interval, log amount and log amount/repeat failures. The existing fitted scaler and encoder transform those features before XGBoost scoring. The model and classification threshold are unchanged.

The scorer is cached as a Streamlit resource. Optimized vectorized inference runs in blocks of up to 8,192 and returns compact 32-bit score arrays. Threshold comparisons use double precision. Session state retains the validated input and one four-byte score per row. The chart uses 40 bins across the two classes, the result table materializes at most 1,000 rows at a time, and CSV generation is deferred until requested. Clearing, replacing or invalidating an upload clears obsolete predictions. The active upload fingerprint is reused across ordinary widget reruns.

The original public and private workspace separation remains in place. Batch inference sends no data to an explanation provider. Only explicit selection of an individual flagged row can trigger SHAP and case persistence. Public uploaded-row reports and Copilot stay offline. Input text and optional columns never enter the prediction CSV. Existing spreadsheet formula escaping still protects case exports containing reviewer notes. See the [CSV contract and sample download](BATCH_DATASET.md).

## Benchmark method

Local Windows laptop, eight logical cores, roughly 7.84 GiB physical RAM. Synthetic rows repeat the bundled 500-record schema-compatible demo to vary model inputs without using private records. The record-count series contains required columns only. The byte series holds data records at 25,000 and adds a removable, ignored optional text column to reach exact decimal byte sizes. The paired targets use both specified byte size and record count. All generated CSVs lived in an automatically cleaned temporary directory.

Each case ran in a new Python worker, using the packaged model and installed dependencies. Warm vector and chunk timings are medians of three measurements. Read, parse and export are single observations. “Processing” is file read + validation/parse + chunked scoring, excluding network transfer, browser rendering, export and cold model load. The export column measures building UTF-8 output in memory. RSS is Windows resident working set, sampled every ~5 ms inside each phase and ~50 ms by the parent. Peak is the largest sampled RSS for the case, including imports and model baseline. RSS can vary with paging and allocator reuse. Phase memory numbers are indicative rather than an OS-enforced limit.

The runner skipped launches below its memory preflight and stopped a worker after repeated samples below 240 MiB available or 120 seconds. A first attempt at the paired targets was skipped during transient host memory pressure. Both completed on the next attempt with more available RAM.

## Measured results

| Case | CSV bytes | Records | Parse s | Vector s | Chunk s | Warm processing s | Rows/s | Peak MiB | Export s | Export MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rows-1000 | 49,796 | 1,000 | 0.135 | 0.025 | 0.024 | 0.176 | 5,688 | 175.7 | 0.013 | 0.08 |
| rows-5000 | 248,364 | 5,000 | 0.088 | 0.018 | 0.021 | 0.122 | 41,017 | 180.3 | 0.023 | 0.41 |
| rows-10000 | 496,574 | 10,000 | 0.106 | 0.050 | 0.065 | 0.181 | 55,096 | 184.0 | 0.072 | 0.83 |
| rows-25000 | 1,241,204 | 25,000 | 0.148 | 0.058 | 0.088 | 0.246 | 101,420 | 198.1 | 0.106 | 2.09 |
| rows-50000 | 2,482,254 | 50,000 | 0.180 | 0.100 | 0.170 | 0.361 | 138,313 | 207.6 | 0.217 | 4.20 |
| rows-100000 | 4,964,354 | 100,000 | 0.386 | 0.235 | 0.321 | 0.721 | 138,735 | 265.0 | 0.518 | 8.40 |
| rows-250000 | 12,410,654 | 250,000 | 0.832 | 0.548 | 0.746 | 1.595 | 156,691 | 374.5 | 1.060 | 21.17 |
| bytes-2MB | 2,000,000 | 25,000 | 0.182 | 0.071 | 0.083 | 0.277 | 90,416 | 198.7 | 0.104 | 2.09 |
| bytes-5MB | 5,000,000 | 25,000 | 0.297 | 0.075 | 0.101 | 0.415 | 60,255 | 204.3 | 0.147 | 2.09 |
| bytes-10MB | 10,000,000 | 25,000 | 0.321 | 0.154 | 0.148 | 0.484 | 51,674 | 209.7 | 0.224 | 2.09 |
| bytes-25MB | 25,000,000 | 25,000 | 0.502 | 0.061 | 0.112 | 0.635 | 39,376 | 224.7 | 0.120 | 2.09 |
| bytes-50MB | 50,000,000 | 25,000 | 0.839 | 0.072 | 0.141 | 1.008 | 24,799 | 272.2 | 0.133 | 2.09 |
| bytes-100MB | 100,000,000 | 25,000 | 1.728 | 0.069 | 0.097 | 1.880 | 13,301 | 387.0 | 0.128 | 2.09 |
| bytes-150MB | 150,000,000 | 25,000 | 2.549 | 0.059 | 0.089 | 2.720 | 9,189 | 546.2 | 0.119 | 2.09 |
| target-10MB-50k | 10,000,000 | 50,000 | 0.409 | 0.120 | 0.185 | 0.610 | 82,034 | 230.1 | 0.225 | 4.20 |
| local-25MB-100k | 25,000,000 | 100,000 | 0.680 | 0.245 | 0.330 | 1.033 | 96,805 | 283.6 | 0.471 | 8.40 |

| Simultaneous sessions | Input per session | Warm wall s, including export | Total rows/s | Shared peak MiB | Retained input, score and output MiB |
|---:|---|---:|---:|---:|---:|
| 2 | 10 MB / 50,000 rows | 1.402 | 71,306 | 243.3 | 29.7 |
| 4 | 10 MB / 50,000 rows | 2.660 | 75,174 | 301.5 | 59.5 |

The synthetic sample repeats known category strings. Real uploads with high-cardinality Unicode category values can use more memory. Optional text width and quoting may also affect parse times.

## Interpretation and deployment gate

At 50,000 narrow records, the emulated legacy parse-and-score path took 3.645 s. The optimized parse-and-score path took 0.350 s, about 10.4× faster for this case. The legacy comparison strips its original hard caps and covers repeated preprocessing and dictionary creation. It excludes legacy dashboard rendering and export.

The 50,000-record compact input used approximately 1.20 MB and its scores 0.20 MB. The corresponding legacy prediction DataFrame alone occupied about 2.05 MB. All headline figures are decimal MB for files or explicitly MiB for resident memory.

At 250,000 records, vector scoring took 0.548 s versus 0.746 s in blocks. The sampled inference-stage RSS was approximately 375 MiB vectorized and 227 MiB chunked. Chunked vectorization is the better default for a shared service because it bounds the dense transformed matrix. All single-file runs produced identical scores and flags for both inference methods, and the five legacy comparison runs also matched exactly.

The local 10 MB / 50,000-record benchmark processed in approximately 0.61 s warm, plus 0.23 s to generate a 4.20 MB export. Two and four peer sessions at that benchmark size were exercised with a shared scorer. The four-session worker peaked near 301 MiB and retained about 59.5 MiB of separate upload, frame, score and download buffers. The cold model load took about 1.7–2.4 s in the paired and concurrency runs. Browser upload, app startup, websocket/Arrow rendering and cold SHAP can add latency.

Streamlit keeps uploads in server memory. Deferred downloads save render-time memory but still allocate the result when requested. The local concurrency simulation retained one upload and one export buffer per session. It excluded actual multipart and transport copies, full Streamlit and SHAP resident memory, and other hosted sessions. Keep headroom for those allocations when observing the deployed app. [Streamlit upload storage](https://docs.streamlit.io/knowledge-base/using-streamlit/where-file-uploader-store-when-deleted) · [Deferred downloads](https://docs.streamlit.io/develop/api-reference/widgets/st.download_button)

Community Cloud publishes approximate, variable resource allocations. These Windows laptop measurements establish local behavior. They provide no hosted throughput or concurrency guarantee. [Community Cloud resources](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

## Reproduce locally

```powershell
python -m pip install psutil==7.2.2
python -m benchmarks.batch_capacity
python -m benchmarks.batch_capacity --targets
python -m pytest -q
```

Raw measurements are available in [the row, size and concurrency results](reports/batch-capacity.json) and [the paired-limit results](reports/batch-capacity-targets.json).

The release regression suite passed **123 tests**. It includes the 122 benchmark-stage tests plus exact acceptance at 5,000,000 bytes and 25,000 records, independent rejection one byte or one row over, full-batch CSV export, and vector/block score parity. Four dependency deprecation warnings remain.
