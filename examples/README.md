# Folate Examples

Runnable examples and live browser tests demonstrating how to use `FolateSelectorFinder` across static DOM snapshots and live Playwright sessions.

## Prerequisites

Install runtime dependencies and Playwright (for live browser tests):

```bash
pip install -r ../requirements.txt
pip install playwright
playwright install chromium
```

## Included Scripts

| Script | Description |
| :--- | :--- |
| `benchmark.py` | Measures model load time, CPU inference latency, process RSS memory, and top-k ranking on a sample travel booking DOM. |
| `test_google.py` | Opens `google.com` in Playwright, finds the search box (`textarea[name="q"]`), types a query, and submits it. |
| `test_amazon.py` | Tests e-commerce element grounding on `amazon.com` (search bar, search submit button, cart link, and department navigation) with offline HTML fallback. |
| `test_github.py` | Navigates to `github.com/figranium/folate` in Playwright and locates repository actions (code/clone menu, issues tab, pull requests tab, and repository search). |
| `test_multistep_checkout.py` | Offline multi-step e-commerce form & checkout flow benchmark (size selection, add to cart, shipping address fields, promo code, and place order). |

## Running

Run any script from the repository root:

```bash
python examples/benchmark.py
python examples/test_google.py
python examples/test_amazon.py
python examples/test_github.py
python examples/test_multistep_checkout.py
```

By default, live browser examples launch Chromium in headed mode (`headless=False`) so you can visually watch Folate select and interact with elements. Pass `--headless` to run without a visible browser window:

```bash
python examples/test_google.py --headless
```
