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
| `test_google.py` | Opens `google.com` in Playwright, finds the search box (`textarea[name="q"]`), types a query, and submits it. |
| `test_amazon.py` | Tests e-commerce element grounding on `amazon.com` (search bar, search submit button, cart link, and department navigation) with offline HTML fallback. |
| `test_github.py` | Navigates to `github.com/figranium/folate` in Playwright and locates repository actions (code/clone menu, issues tab, pull requests tab, and repository search). |

## Running Individual Scripts

```bash
python examples/test_google.py
python examples/test_amazon.py --offline
python examples/test_github.py
```
