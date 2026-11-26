"""Local smoke timings. These results are not Community Cloud measurements."""

from concurrent.futures import ThreadPoolExecutor
import json
import sys
from time import perf_counter

import numpy as np
import psutil

from odor_app.artifacts import BUNDLE, load_bundle, manifest_digest
from odor_app.inference import Predictor


def peak_rss_mib(process):
    if sys.platform == "win32":
        return process.memory_info().peak_wset / 2**20
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**20 if sys.platform == "darwin" else 1024)


def run():
    process = psutil.Process()
    start = perf_counter()
    predictor = Predictor(load_bundle(BUNDLE, manifest_digest()))
    load_seconds = perf_counter() - start

    def one(smiles):
        start = perf_counter()
        prediction = predictor.predict(smiles)
        return (perf_counter() - start) * 1000, prediction.scores

    example = "CC(=O)OCC"
    baseline = one(example)[1]
    timings = [one(example)[0] for _ in range(30)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(one, [example] * 8))
    assert all(scores == baseline for _, scores in parallel)
    rows = [predictor.features.transform(example) for _ in range(100)]
    start = perf_counter()
    assert len(predictor.predict_many(rows)) == 100
    print(json.dumps({"load_seconds": round(load_seconds, 3), "warm_p95_ms": round(float(np.percentile(timings, 95)), 2),
                      "batch_100_predict_seconds": round(perf_counter() - start, 3),
                      "rss_after_tests_mib": round(process.memory_info().rss / 2**20, 1),
                      "peak_rss_mib": round(peak_rss_mib(process), 1),
                      "two_concurrent_workers_consistent": True}, indent=2))


if __name__ == "__main__":
    run()
