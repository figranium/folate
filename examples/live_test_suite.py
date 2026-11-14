import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
import psutil
from playwright.async_api import async_playwright

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from folate import FolateSelectorFinder  # noqa: E402


@dataclass
class TestStep:
    query: str
    expected_selectors: list[str]
    expected_text_any: list[str] = field(default_factory=list)
    action: str | None = None  # "click" | "fill" | "select" | None
    value: str | None = None
    press_enter: bool = False
    wait_after_ms: int = 1500


@dataclass
class SiteSuite:
    name: str
    url: str
    steps: list[TestStep]
    dismiss_queries: list[str] = field(default_factory=list)
    bot_block_markers: list[str] = field(default_factory=list)


LIVE_SUITES: dict[str, SiteSuite] = {
    "google": SiteSuite(
        name="Google Search",
        url="https://www.google.com",
        dismiss_queries=["click Reject all cookies button", "click Accept all cookies button"],
        bot_block_markers=["/sorry/index", "unusual traffic from your computer"],
        steps=[
            TestStep(
                query="type search query in Google search box",
                expected_selectors=['textarea[name="q"]', 'input[name="q"]'],
                action="fill",
                value="Figranium browser automation",
                press_enter=True,
                wait_after_ms=2500,
            ),
            TestStep(
                query="click Images tab",
                expected_selectors=['a[href*="tbm=isch"]', 'a[href*="udm=2"]'],
                expected_text_any=["images"],
                action=None,
            ),
        ],
    ),
    "amazon": SiteSuite(
        name="Amazon Storefront",
        url="https://www.amazon.com",
        bot_block_markers=["validateCaptcha", "Type the characters you see in this image"],
        steps=[
            TestStep(
                query="type product name in Amazon search box",
                expected_selectors=["#twotabsearchtextbox", 'input[name="field-keywords"]'],
                action="fill",
                value="wireless mechanical keyboard",
                press_enter=False,
                wait_after_ms=400,
            ),
            TestStep(
                query="click search submit button",
                expected_selectors=["#nav-search-submit-button"],
                action=None,
            ),
            TestStep(
                query="open shopping cart",
                expected_selectors=["#nav-cart", 'a[href*="/cart"]'],
                expected_text_any=["cart"],
                action=None,
            ),
            TestStep(
                query="click Today's Deals link",
                expected_selectors=['a[href*="deals"]', 'a[href*="goldbox"]'],
                expected_text_any=["deals"],
                action="click",
                wait_after_ms=2000,
            ),
        ],
    ),
    "github": SiteSuite(
        name="GitHub Repository (figranium/folate)",
        url="https://github.com/figranium/folate",
        steps=[
            TestStep(
                query="click Code button to clone repository",
                expected_selectors=["button"],
                expected_text_any=["code"],
                action=None,
            ),
            TestStep(
                query="click README.md file link",
                expected_selectors=['a[href*="README.md"]'],
                expected_text_any=["readme.md"],
                action=None,
            ),
            TestStep(
                query="click Pull requests tab",
                expected_selectors=['a[href="/figranium/folate/pulls"]'],
                expected_text_any=["pull requests"],
                action=None,
            ),
            TestStep(
                query="click Issues tab",
                expected_selectors=['a[href="/figranium/folate/issues"]'],
                expected_text_any=["issues"],
                action="click",
                wait_after_ms=2000,
            ),
        ],
    ),
    "wikipedia": SiteSuite(
        name="Wikipedia (Main Page -> Search -> Article)",
        url="https://en.wikipedia.org/wiki/Main_Page",
        steps=[
            TestStep(
                query="type topic in Wikipedia search box",
                expected_selectors=['input[name="search"]', "#searchInput"],
                action="fill",
                value="Artificial neural network",
                press_enter=True,
                wait_after_ms=2000,
            ),
            TestStep(
                query="click Talk discussion tab",
                expected_selectors=['a[href*="Talk:"]', "#ca-talk a"],
                expected_text_any=["talk"],
                action=None,
            ),
            TestStep(
                query="click View history tab",
                expected_selectors=['a[href*="action=history"]', "#ca-history a"],
                expected_text_any=["history"],
                action="click",
                wait_after_ms=2000,
            ),
        ],
    ),
    "hackernews": SiteSuite(
        name="Hacker News",
        url="https://news.ycombinator.com",
        steps=[
            TestStep(
                query="click new stories link in top navigation",
                expected_selectors=['a[href="newest"]'],
                expected_text_any=["new"],
                action="click",
                wait_after_ms=1500,
            ),
            TestStep(
                query="click ask HN link",
                expected_selectors=['a[href="ask"]'],
                expected_text_any=["ask"],
                action="click",
                wait_after_ms=1500,
            ),
            TestStep(
                query="click login link",
                expected_selectors=['a[href*="login"]'],
                expected_text_any=["login"],
                action=None,
            ),
        ],
    ),
    "duckduckgo": SiteSuite(
        name="DuckDuckGo Search",
        url="https://duckduckgo.com",
        steps=[
            TestStep(
                query="type search query in DuckDuckGo search box",
                expected_selectors=['input[name="q"]', "#searchbox_input"],
                action="fill",
                value="ONNX Runtime INT8 quantization",
                press_enter=True,
                wait_after_ms=2500,
            ),
            TestStep(
                query="click Images search tab",
                expected_selectors=['a[href*="iax=images"]', 'a[data-zci-link="images"]'],
                expected_text_any=["images"],
                action=None,
            ),
        ],
    ),
    "python_org": SiteSuite(
        name="Python.org Documentation & Downloads",
        url="https://www.python.org",
        steps=[
            TestStep(
                query="type package or topic in Python.org search box",
                expected_selectors=["#id-search-field", 'input[name="q"]'],
                action="fill",
                value="asyncio",
                press_enter=False,
                wait_after_ms=300,
            ),
            TestStep(
                query="click GO search button",
                expected_selectors=["#submit", 'button[name="submit"]'],
                expected_text_any=["go"],
                action=None,
            ),
            TestStep(
                query="click Downloads navigation link",
                expected_selectors=['a[href="/downloads/"]'],
                expected_text_any=["downloads"],
                action="click",
                wait_after_ms=2000,
            ),
        ],
    ),
    "mdn": SiteSuite(
        name="MDN Web Docs",
        url="https://developer.mozilla.org/en-US/",
        steps=[
            TestStep(
                query="click References navigation menu",
                expected_selectors=['a[href*="/docs/Web"]'],
                expected_text_any=["references"],
                action="click",
                wait_after_ms=2000,
            ),
            TestStep(
                query="click HTML reference link",
                expected_selectors=['a[href*="/docs/Web/HTML"]'],
                expected_text_any=["html"],
                action="click",
                wait_after_ms=2000,
            ),
        ],
    ),
    "quotes_sandbox": SiteSuite(
        name="Quotes to Scrape (Multi-Step Login & Pagination)",
        url="https://quotes.toscrape.com",
        steps=[
            TestStep(
                query="click Login link",
                expected_selectors=['a[href="/login"]'],
                expected_text_any=["login"],
                action="click",
                wait_after_ms=1500,
            ),
            TestStep(
                query="type username in login form",
                expected_selectors=["#username", 'input[name="username"]'],
                action="fill",
                value="figranium_tester",
                wait_after_ms=300,
            ),
            TestStep(
                query="type password in login form",
                expected_selectors=["#password", 'input[name="password"]'],
                action="fill",
                value="hunter2",
                wait_after_ms=300,
            ),
            TestStep(
                query="click Login submit button",
                expected_selectors=['input[type="submit"]', "button"],
                expected_text_any=["login"],
                action="click",
                wait_after_ms=1500,
            ),
            TestStep(
                query="click Next page pagination link",
                expected_selectors=['a[href="/page/2/"]'],
                expected_text_any=["next"],
                action="click",
                wait_after_ms=1500,
            ),
        ],
    ),
    "books_sandbox": SiteSuite(
        name="Books to Scrape (Category & Product Navigation)",
        url="https://books.toscrape.com",
        steps=[
            TestStep(
                query="click Travel category link in sidebar",
                expected_selectors=['a[href*="travel_2"]'],
                expected_text_any=["travel"],
                action="click",
                wait_after_ms=1500,
            ),
            TestStep(
                query="click It's Only the Himalayas book link",
                expected_selectors=['a[href*="its-only-the-himalayas"]'],
                expected_text_any=["himalayas"],
                action="click",
                wait_after_ms=1500,
            ),
        ],
    ),
}


def resolve_locator(page, selector: str):
    if selector.startswith("/"):
        return page.locator(f"xpath={selector}").first
    return page.locator(selector).first


def match_matches_expectation(match: dict, step: TestStep) -> bool:
    pred_sel = match["selector"].lower()
    pred_el = match["element"].lower()

    sel_ok = any(exp.lower() in pred_sel or pred_sel in exp.lower() for exp in step.expected_selectors)
    if step.expected_text_any:
        text_ok = any(tok.lower() in pred_el for tok in step.expected_text_any)
        return sel_ok and text_ok
    return sel_ok


async def dismiss_interstitials(page, finder: FolateSelectorFinder, queries: list[str]):
    if not queries:
        return
    html = await page.content()
    for q in queries:
        matches = finder.rank(q, html, top_k=1)
        if matches and matches[0]["score"] >= 0.60:
            try:
                loc = resolve_locator(page, matches[0]["selector"])
                if await loc.is_visible(timeout=800):
                    await loc.click(timeout=1500)
                    await page.wait_for_timeout(1000)
                    break
            except Exception:
                pass


async def execute_step_action(page, selector: str, step: TestStep):
    if not step.action:
        return
    loc = resolve_locator(page, selector)
    await loc.wait_for(state="visible", timeout=5000)
    if step.action == "fill":
        await loc.fill(step.value or "", timeout=5000)
        if step.press_enter:
            await page.keyboard.press("Enter")
    elif step.action == "click":
        await loc.click(timeout=5000)
    elif step.action == "select":
        await loc.select_option(step.value or "", timeout=5000)
    if step.wait_after_ms > 0:
        await page.wait_for_timeout(step.wait_after_ms)


async def run_site_suite(
    context,
    finder: FolateSelectorFinder,
    site_key: str,
    suite: SiteSuite,
    top_k: int = 3,
    max_candidates: int = 48,
) -> dict:
    page = await context.new_page()
    site_report = {
        "site": site_key,
        "name": suite.name,
        "url": suite.url,
        "status": "completed",
        "steps": [],
    }
    print(f"\n{'=' * 78}\n[{site_key.upper()}] {suite.name} -> {suite.url}\n{'=' * 78}")

    try:
        await page.goto(suite.url, wait_until="domcontentloaded", timeout=20000)
        await page.wait_for_timeout(1500)
        await dismiss_interstitials(page, finder, suite.dismiss_queries)

        initial_html = await page.content()
        if any(marker.lower() in initial_html.lower() for marker in suite.bot_block_markers):
            print("  [SKIP] Bot-protection interstitial detected on live site.")
            site_report["status"] = "skipped_bot_block"
            return site_report

        for idx, step in enumerate(suite.steps, 1):
            html = await page.content()
            total_cands = len(finder.extract_candidates(html))

            t0 = time.perf_counter()
            matches = finder.rank(
                step.query,
                html,
                top_k=top_k,
                max_candidates=max_candidates,
            )
            infer_ms = (time.perf_counter() - t0) * 1000

            top1_pass = bool(matches) and match_matches_expectation(matches[0], step)
            topk_pass = any(match_matches_expectation(m, step) for m in matches)
            badge = "PASS" if top1_pass else ("TOP-K" if topk_pass else "FAIL")

            print(
                f"  [{badge:5s}] Step {idx:02d} ({infer_ms:6.1f} ms | {total_cands:3d} DOM nodes) | "
                f"Query: '{step.query}'"
            )
            for rank_i, m in enumerate(matches, 1):
                print(f"          #{rank_i} ({m['score']:.4f}) -> {m['selector']} | {m['element'][:95]}")

            action_error = None
            if matches and step.action:
                try:
                    await execute_step_action(page, matches[0]["selector"], step)
                except Exception as exc:
                    action_error = str(exc).splitlines()[0]
                    print(f"          [WARN] Action execution error: {action_error}")

            site_report["steps"].append(
                {
                    "step": idx,
                    "query": step.query,
                    "dom_candidates": total_cands,
                    "infer_ms": round(infer_ms, 2),
                    "top1_pass": top1_pass,
                    "topk_pass": topk_pass,
                    "predictions": matches,
                    "action_error": action_error,
                }
            )
    except Exception as exc:
        err = str(exc).splitlines()[0]
        print(f"  [ERROR] Suite aborted: {err}")
        site_report["status"] = f"error: {err}"
    finally:
        await page.close()

    return site_report


async def main():
    parser = argparse.ArgumentParser(description="Complete live Playwright + Folate end-to-end test suite.")
    parser.add_argument(
        "--sites",
        nargs="+",
        default=list(LIVE_SUITES.keys()),
        choices=list(LIVE_SUITES.keys()),
        help="Subset of live site suites to run (default: all 10 suites)",
    )
    parser.add_argument("--headless", action="store_true", help="Run Chromium in headless mode")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K predictions to evaluate per step")
    parser.add_argument("--max-candidates", type=int, default=48, help="Max pre-filtered DOM candidates per step")
    parser.add_argument("--model-dir", default=None, help="Optional local model directory override")
    parser.add_argument("--json-out", default=None, help="Optional path to save detailed JSON benchmark report")
    args = parser.parse_args()

    proc = psutil.Process(os.getpid())
    t_load = time.perf_counter()
    finder = FolateSelectorFinder(model_dir=args.model_dir)
    load_ms = (time.perf_counter() - t_load) * 1000

    print(f"Loaded Folate ONNX engine in {load_ms:.1f} ms | Initial RSS: {proc.memory_info().rss / (1024 * 1024):.1f} MB")

    reports = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=args.headless)
        context = await browser.new_context(
            viewport={"width": 1366, "height": 860},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        for site_key in args.sites:
            report = await run_site_suite(
                context=context,
                finder=finder,
                site_key=site_key,
                suite=LIVE_SUITES[site_key],
                top_k=args.top_k,
                max_candidates=args.max_candidates,
            )
            reports.append(report)
        await browser.close()

    all_steps = [s for r in reports for s in r["steps"]]
    total_steps = len(all_steps)
    top1_hits = sum(1 for s in all_steps if s["top1_pass"])
    topk_hits = sum(1 for s in all_steps if s["topk_pass"])
    avg_ms = (sum(s["infer_ms"] for s in all_steps) / total_steps) if total_steps else 0.0
    peak_ram_mb = proc.memory_info().rss / (1024 * 1024)

    print(f"\n{'=' * 78}")
    print("LIVE SUITE SUMMARY")
    print(f"{'=' * 78}")
    print(f"Sites Completed : {sum(1 for r in reports if r['status'] == 'completed')}/{len(reports)}")
    if total_steps:
        print(f"Top-1 Accuracy  : {top1_hits}/{total_steps} ({top1_hits / total_steps * 100:.1f}%)")
        print(f"Top-{args.top_k} Recall    : {topk_hits}/{total_steps} ({topk_hits / total_steps * 100:.1f}%)")
        print(f"Avg CPU Latency : {avg_ms:.1f} ms / step")
    print(f"Python RSS RAM  : {peak_ram_mb:.1f} MB")

    if args.json_out:
        summary_payload = {
            "model_load_ms": round(load_ms, 2),
            "python_rss_mb": round(peak_ram_mb, 2),
            "total_steps": total_steps,
            "top1_hits": top1_hits,
            "topk_hits": topk_hits,
            "avg_infer_ms": round(avg_ms, 2),
            "suites": reports,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"Saved JSON report to {args.json_out}")


if __name__ == "__main__":
    asyncio.run(main())
