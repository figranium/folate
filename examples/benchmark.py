import os
import sys
import time
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from folate import FolateSelectorFinder  # noqa: E402

SAMPLE_HTML = """
<html><body>
  <nav>
    <a href="/flights">Flights</a>
    <a href="/hotels">Stays & Hotels</a>
    <button id="sign-in-btn" aria-label="Sign in to account">Sign in</button>
  </nav>
  <form>
    <input type="text" name="origin" placeholder="Where from?" />
    <input type="text" name="destination" placeholder="Where to?" />
    <button type="submit" data-testid="search-flights">Search Flights</button>
  </form>
</body></html>
"""


def main():
    model_dir = os.environ.get("FOLATE_MODEL_DIR", "./folate-int8")
    proc = psutil.Process(os.getpid())

    t0 = time.perf_counter()
    finder = FolateSelectorFinder(model_dir=model_dir)
    load_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    matches = finder.rank("click Search Flights button", SAMPLE_HTML, top_k=3)
    infer_ms = (time.perf_counter() - t1) * 1000
    ram_mb = proc.memory_info().rss / (1024 * 1024)

    print(f"Model Load: {load_ms:.1f} ms | Inference (6 candidates): {infer_ms:.1f} ms | Process RAM: {ram_mb:.1f} MB\n")
    for rank_i, m in enumerate(matches, 1):
        print(f"#{rank_i} ({m['score']:.4f}) -> {m['selector']} | {m['element']}")


if __name__ == "__main__":
    main()
