"""Isolated, local CSV capacity benchmark. Run: python -m benchmarks.batch_capacity."""
import concurrent.futures
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from time import perf_counter, sleep

import psutil

ROOT = Path(__file__).resolve().parents[1]
MIB = 1048576
FLOOR = 240 * MIB


def make_file(path, rows, size=None):
    source = (ROOT / "data/demo.csv").read_bytes().splitlines()
    header = b",".join(source[0].split(b",")[1:])
    samples = [line.split(b",", 1)[1] for line in source[1:]]
    lines = [samples[i % len(samples)] for i in range(rows)]
    padding_width = remainder = 0
    if size:
        header += b",optional_reference"
        padding_width, remainder = divmod(size - len(header) - 1 - sum(len(line)+2 for line in lines), rows)
        if padding_width < 0:
            raise ValueError("Requested byte size too small")
    padding = b"x" * padding_width
    with path.open("wb") as out:
        out.write(header + b"\n")
        for i, line in enumerate(lines):
            if size:
                out.write(line + b"," + padding + (b"x" if i < remainder else b"") + b"\n")
            else:
                out.write(line + b"\n")


def worker(path, repetitions, clients, legacy):
    import io
    import csv

    import numpy as np
    import pandas as pd

    from fraudshield.batch import export_predictions, read_upload, score_upload
    from fraudshield.features import FIELDS, features
    from fraudshield.service import Scorer

    # A cold model is loaded once, as in the public Streamlit process.
    start = perf_counter()
    scorer = Scorer()
    model_load = perf_counter() - start
    proc = psutil.Process()
    idle = proc.memory_info().rss

    def measured(fn, n=1):
        timings, peak = [], [proc.memory_info().rss]
        done = threading.Event()

        def sample():
            while not done.wait(.005):
                peak[0] = max(peak[0], proc.memory_info().rss)

        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        try:
            for _ in range(n):
                t = perf_counter()
                result = fn()
                timings.append(perf_counter()-t)
                peak[0] = max(peak[0], proc.memory_info().rss)
        finally:
            done.set()
            monitor.join()
        return result, round(statistics.median(timings), 4), peak[0]

    def parse(raw):
        return read_upload(raw, max_bytes=151_000_000, max_rows=250_000)

    if clients > 1:
        # Model shared, uploads, parses, scores and download bytes separate per simulated session.
        def session(_):
            raw = path.read_bytes()
            frame = parse(raw)
            scores = score_upload(scorer, frame)
            download = export_predictions(scores, scorer.metadata).encode()
            return raw, frame, scores, download

        def concurrent_sessions():
            with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as pool:
                return list(pool.map(session, range(clients)))

        results, wall, peak = measured(concurrent_sessions)
        assert all(np.array_equal(results[0][2], result[2]) for result in results)
        retained = sum(len(raw)+int(frame.memory_usage(deep=True).sum())+scores.nbytes+len(download)
                       for raw, frame, scores, download in results)
        return dict(clients=clients, model_load_s=model_load, idle_rss=idle, total_s=wall, peak_rss=peak,
                    retained_bytes=retained, rows=len(results[0][1]), consistent=True)

    raw, read_s, read_rss = measured(path.read_bytes)
    frame, parse_s, parse_rss = measured(lambda: parse(raw))
    chunked, chunk_s, chunk_rss = measured(lambda: score_upload(scorer, frame), repetitions)
    vector, vector_s, vector_rss = measured(lambda: score_upload(scorer, frame, chunk_size=len(frame)), repetitions)
    result = dict(rows=len(frame), model_load_s=model_load, idle_rss=idle,
                  file_read_s=read_s, file_read_rss=read_rss, parse_s=parse_s, parse_rss=parse_rss,
                  vector_s=vector_s, vector_rss=vector_rss, chunk_s=chunk_s, chunk_rss=chunk_rss,
                  frame_bytes=int(frame.memory_usage(deep=True).sum()), score_bytes=chunked.nbytes,
                  exact_score_match=bool(np.array_equal(chunked, vector)),
                  exact_flag_match=bool(np.array_equal(chunked.astype(float) >= scorer.metadata["threshold"],
                                                      vector.astype(float) >= scorer.metadata["threshold"])))
    assert result["exact_score_match"] and result["exact_flag_match"]
    output, export_s, export_rss = measured(lambda: export_predictions(chunked, scorer.metadata).encode())
    result.update(export_s=export_s, export_rss=export_rss, export_bytes=len(output))
    del output, vector
    if legacy:
        # Reproduce the previous scaled path without its fixed byte and row caps.
        def old_path():
            with io.StringIO(raw.decode("utf-8-sig")) as text:
                records = csv.reader(text, strict=True)
                next(records)
                assert sum(bool(row) for row in records) == len(frame)
            original = pd.read_csv(io.BytesIO(raw), usecols=FIELDS)
            features(original)
            rows = []
            for start in range(0, len(original), 250):
                rows.extend(scorer.batch(original.iloc[start:start+250]))
            return pd.DataFrame(rows)

        old, legacy_s, legacy_rss = measured(old_path)
        assert np.array_equal(old.risk.to_numpy(), chunked)
        assert np.array_equal(old.suspected_fraud.to_numpy(), chunked.astype(float) >= scorer.metadata["threshold"])
        result.update(legacy_parse_and_score_s=legacy_s, legacy_rss=legacy_rss,
                      legacy_score_match=True, legacy_result_bytes=int(old.memory_usage(deep=True).sum()))
    return result


def main(targets=False):
    report = {"platform": platform.platform(), "logical_cores": psutil.cpu_count(),
              "system_memory": psutil.virtual_memory().total, "minimum_available_bytes": FLOOR, "cases": []}
    destination = ROOT / "reports" / ("batch-capacity-targets.json" if targets else "batch-capacity.json")
    cases = [(f"rows-{rows}", rows, None, 1) for rows in (1000, 5000, 10000, 25000, 50000, 100000, 250000)]
    cases += [(f"bytes-{mb}MB", 25000, mb * 1_000_000, 1) for mb in (2, 5, 10, 25, 50, 100, 150)]
    cases += [(f"concurrent-{clients}", 50000, 10_000_000, clients) for clients in (2, 4)]
    if targets:
        cases = [("target-10MB-50k", 50000, 10_000_000, 1),
                 ("local-25MB-100k", 100000, 25_000_000, 1)]
    with tempfile.TemporaryDirectory(prefix="fraudshield-batch-audit-") as temp:
        temp = Path(temp)
        for name, rows, size, clients in cases:
            available = psutil.virtual_memory().available
            case = {"case": name, "planned_rows": rows, "available_at_start": available}
            if available < FLOOR + 200 * MIB:
                case["skipped"] = "Insufficient available RAM before launch"
            else:
                path = temp / "synthetic.csv"
                make_file(path, rows, size)
                case["file_bytes"] = path.stat().st_size
                out = temp / "worker.json"
                err = temp / "worker.err"
                with out.open("wb") as stdout, err.open("wb") as stderr:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "benchmarks.batch_capacity", str(path), str(clients),
                         str(int(name.startswith("rows") and rows <= 50000))],
                        cwd=ROOT, stdout=stdout, stderr=stderr,
                    )
                    peak = 0
                    low = 0
                    started = perf_counter()
                    minimum = available
                    while process.poll() is None:
                        free = psutil.virtual_memory().available
                        minimum = min(minimum, free)
                        low = low + 1 if free < FLOOR else 0
                        try:
                            peak = max(peak, psutil.Process(process.pid).memory_info().rss)
                        except psutil.NoSuchProcess:
                            pass
                        if low >= 4 or perf_counter() - started > 120:
                            process.terminate()
                            case["skipped"] = "Host memory or elapsed-time guard stopped worker"
                            break
                        sleep(.05)
                    process.wait()
                case.update(worker_peak_rss=peak, minimum_available=minimum)
                if process.returncode == 0:
                    case.update(json.loads(out.read_text()))
                elif "skipped" not in case:
                    case["failed"] = err.read_text(errors="replace").splitlines()[-1]
            report["cases"].append(case)
            print(name, case.get("skipped", case.get("failed", "complete")), flush=True)
            destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if "guard stopped" in case.get("skipped", ""):
                report["stopped_after"] = name
                destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
                break


if __name__ == "__main__":
    if sys.argv[1:] == ["--targets"]:
        main(targets=True)
    elif len(sys.argv) > 1:
        print(json.dumps(worker(Path(sys.argv[1]), 3, int(sys.argv[2]), bool(int(sys.argv[3])))))
    else:
        main()
