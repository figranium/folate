# Folate Examples

Runnable examples and live browser tests demonstrating how to use `FolateSelectorFinder` across static DOM snapshots and live Playwright sessions.

## Prerequisites

Install runtime dependencies and Playwright (for live browser tests):

```bash
python -m pip install -r ../requirements.txt
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Included Scripts

| Script | Description |
| :--- | :--- |
| `live_test_suite.py` | **Complete 10-site, 31-step live Playwright benchmark suite** covering Google, Amazon, GitHub, Wikipedia, Hacker News, DuckDuckGo, Python.org, MDN Web Docs, Quotes to Scrape (multi-step login + pagination), and Books to Scrape—reporting Top-1 accuracy, Top-K recall, per-step CPU latency, RSS RAM, and optional JSON output. |
| `benchmark.py` | Measures model load time, CPU inference latency, process RSS memory, and top-k ranking on a sample travel booking DOM. |
| `test_google.py` | Opens `google.com` in Playwright, finds the search box (`textarea[name="q"]`), types a query, and submits it. |
| `test_amazon.py` | Tests e-commerce element grounding on `amazon.com` (search bar, search submit button, cart link, and department navigation) with offline HTML fallback. |
| `test_github.py` | Navigates to `github.com/figranium/folate` in Playwright and locates repository actions (code/clone menu, issues tab, pull requests tab, and repository search). |
| `test_multistep_checkout.py` | Offline multi-step e-commerce form & checkout flow benchmark (size selection, add to cart, shipping address fields, promo code, and place order). |

## Running the Complete Live Test Suite

Run all 10 live site suites end-to-end:

```bash
python examples/live_test_suite.py
```

Run headless, target specific sites, or export a JSON report:

```bash
python examples/live_test_suite.py --headless --sites google github wikipedia hackernews --json-out live_results.json
```

## Running Individual Scripts

```bash
python examples/benchmark.py
python examples/test_multistep_checkout.py
python examples/test_google.py
python examples/test_amazon.py --offline
python examples/test_github.py
```
