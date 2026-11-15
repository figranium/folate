import argparse
import json
import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from folate import FolateSelectorFinder  # noqa: E402


DEFAULT_FIXTURES = [
    {
        "id": "nav_signin_button",
        "query": "click Sign in button",
        "gt_selector": "#sign-in-btn",
        "html": """
        <html><body>
          <nav>
            <a href="/flights">Flights</a>
            <a href="/hotels">Stays &amp; Hotels</a>
            <button id="sign-in-btn" aria-label="Sign in to account">Sign in</button>
          </nav>
          <main>
            <button class="promo-btn">Sign in for rewards</button>
          </main>
        </body></html>
        """,
    },
    {
        "id": "form_search_flights",
        "query": "click Search Flights button",
        "gt_selector": "button[data-testid='search-flights']",
        "html": """
        <html><body>
          <form id="flight-search">
            <input type="text" name="origin" placeholder="Where from?" />
            <input type="text" name="destination" placeholder="Where to?" />
            <button type="submit" data-testid="search-flights">Search Flights</button>
          </form>
        </body></html>
        """,
    },
    {
        "id": "checkout_email_input",
        "query": "enter email address for receipt",
        "gt_selector": "#email-addr",
        "html": """
        <html><body>
          <form id="shipping-form">
            <input type="text" id="full-name" name="fullName" placeholder="Full name" />
            <input type="email" id="email-addr" name="email" placeholder="Email address for receipt" />
            <input type="text" id="postal-code" name="zip" placeholder="ZIP / Postal code" />
            <button type="submit" id="place-order-btn">Place Order</button>
          </form>
        </body></html>
        """,
    },
    {
        "id": "checkout_apply_promo",
        "query": "click Apply promo button",
        "gt_selector": "#apply-promo",
        "html": """
        <html><body>
          <section aria-label="Discount and Payment">
            <input type="text" id="promo-input" name="promoCode" placeholder="Enter promo or gift code" />
            <button type="button" id="apply-promo">Apply</button>
            <button type="submit" id="place-order-btn" data-testid="submit-order">Place Order</button>
          </section>
        </body></html>
        """,
    },
    {
        "id": "repo_issues_tab",
        "query": "click Issues tab",
        "gt_selector": "a[href='/figranium/folate/issues']",
        "html": """
        <html><body>
          <nav aria-label="Repository">
            <a href="/figranium/folate">Code</a>
            <a href="/figranium/folate/issues">Issues</a>
            <a href="/figranium/folate/pulls">Pull requests</a>
          </nav>
        </body></html>
        """,
    },
]


def resolve_elements(page, selector: str):
    if selector.startswith("/"):
        return page.query_selector_all(f"xpath={selector}")
    return page.query_selector_all(selector)


def load_samples(dataset_path: str | None) -> list[dict]:
    if not dataset_path:
        return DEFAULT_FIXTURES

    path = Path(dataset_path)
    base_dir = path.parent
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "html" not in row and "html_path" in row:
            html_file = Path(row["html_path"])
            if not html_file.is_absolute():
                html_file = base_dir / html_file
            row["html"] = html_file.read_text(encoding="utf-8")
        samples.append(row)
    return samples


def evaluate_dom_precision(
    finder: FolateSelectorFinder,
    samples: list[dict],
    confidence_threshold: float = 0.5,
    max_candidates: int = 48,
) -> dict:
    tp = 0
    fp_wrong = 0
    fp_ambiguous = 0
    fp_broken = 0
    abstained = 0
    total_ms = 0.0
    details = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for idx, row in enumerate(samples, start=1):
            sample_id = row.get("id", f"sample_{idx:03d}")
            html = row["html"]
            query = row["query"]
            gt_selector = row["gt_selector"]

            page.set_content(html, wait_until="domcontentloaded")
            gt_matches = resolve_elements(page, gt_selector)
            if len(gt_matches) != 1:
                raise ValueError(
                    f"Ground-truth selector '{gt_selector}' in '{sample_id}' matched {len(gt_matches)} elements (expected 1)."
                )
            gt_el = gt_matches[0]

            t0 = time.perf_counter()
            matches = finder.rank(query, html, top_k=1, max_candidates=max_candidates)
            infer_ms = (time.perf_counter() - t0) * 1000
            total_ms += infer_ms

            if not matches or matches[0]["score"] < confidence_threshold:
                abstained += 1
                outcome = "abstained"
                pred_sel = matches[0]["selector"] if matches else None
                pred_score = matches[0]["score"] if matches else 0.0
            else:
                pred_sel = matches[0]["selector"]
                pred_score = matches[0]["score"]
                try:
                    pred_matches = resolve_elements(page, pred_sel)
                    if len(pred_matches) == 1 and gt_el.evaluate("(a, b) => a === b", pred_matches[0]):
                        tp += 1
                        outcome = "tp"
                    elif len(pred_matches) > 1:
                        fp_ambiguous += 1
                        outcome = "fp_ambiguous"
                    elif len(pred_matches) == 0:
                        fp_broken += 1
                        outcome = "fp_zero_match"
                    else:
                        fp_wrong += 1
                        outcome = "fp_wrong_node"
                except Exception:
                    fp_broken += 1
                    outcome = "fp_invalid_syntax"

            details.append(
                {
                    "id": sample_id,
                    "query": query,
                    "gt_selector": gt_selector,
                    "pred_selector": pred_sel,
                    "score": round(pred_score, 4),
                    "outcome": outcome,
                    "infer_ms": round(infer_ms, 2),
                }
            )

        browser.close()

    total_samples = len(samples)
    total_predicted = tp + fp_wrong + fp_ambiguous + fp_broken
    precision = (tp / total_predicted * 100) if total_predicted else 0.0
    recall = (tp / total_samples * 100) if total_samples else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    coverage = (total_predicted / total_samples * 100) if total_samples else 0.0

    return {
        "samples": total_samples,
        "confidence_threshold": confidence_threshold,
        "tp": tp,
        "fp_wrong": fp_wrong,
        "fp_ambiguous": fp_ambiguous,
        "fp_broken": fp_broken,
        "abstained": abstained,
        "coverage": round(coverage, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "avg_ms_per_step": round(total_ms / total_samples, 2) if total_samples else 0.0,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure DOM-resolved selector Precision, Recall, and F1 in headless Chromium."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to JSONL file with {id, query, gt_selector, html|html_path} rows (uses built-in fixtures if omitted)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum sigmoid probability required to emit a selector (default: 0.5)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=48,
        help="Max pre-filtered DOM candidates per page (default: 48)",
    )
    parser.add_argument("--model-dir", default=None, help="Optional local model directory override")
    parser.add_argument("--json-out", default=None, help="Optional path to save detailed JSON report")
    args = parser.parse_args()

    samples = load_samples(args.dataset)
    finder = FolateSelectorFinder(model_dir=args.model_dir)
    report = evaluate_dom_precision(
        finder=finder,
        samples=samples,
        confidence_threshold=args.confidence_threshold,
        max_candidates=args.max_candidates,
    )

    print(f"\n{'=' * 64}")
    print("DOM-RESOLVED SELECTOR PRECISION & RECALL REPORT")
    print(f"{'=' * 64}")
    print(f"Samples:              {report['samples']}")
    print(f"Confidence Threshold: {report['confidence_threshold']:.2f}")
    print(f"True Positives (TP):  {report['tp']}")
    print(f"FP (Wrong Node):      {report['fp_wrong']}")
    print(f"FP (Ambiguous >1):    {report['fp_ambiguous']}")
    print(f"FP (Broken / 0):      {report['fp_broken']}")
    print(f"Abstained (FN):       {report['abstained']}")
    print("-" * 64)
    print(f"Coverage:             {report['coverage']:.2f}%")
    print(f"Precision:            {report['precision']:.2f}%")
    print(f"Recall:               {report['recall']:.2f}%")
    print(f"F1 Score:             {report['f1']:.2f}%")
    print(f"Avg CPU Latency:      {report['avg_ms_per_step']:.1f} ms/step")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved detailed report to {args.json_out}")


if __name__ == "__main__":
    main()
