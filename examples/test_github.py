import argparse
import asyncio
import os
import sys
import time
from playwright.async_api import async_playwright

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from folate import FolateSelectorFinder  # noqa: E402

QUERIES = [
    "click Issues tab",
    "click Pull requests tab",
    "click Code button to clone repository",
    "click README.md file link",
]


async def run_test(headless: bool = False):
    finder = FolateSelectorFinder()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page(viewport={"width": 1280, "height": 850})

        await page.goto("https://github.com/figranium/folate", wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)

        html = await page.content()
        candidates = finder.extract_candidates(html)
        print(f"Extracted {len(candidates)} interactive elements from github.com/figranium/folate\n")

        for query in QUERIES:
            t0 = time.perf_counter()
            matches = finder.rank(query, html, top_k=2)
            ms = (time.perf_counter() - t0) * 1000
            print(f"Query: '{query}' ({ms:.1f} ms)")
            for rank_i, m in enumerate(matches, 1):
                print(f"  #{rank_i} ({m['score']:.4f}) -> {m['selector']} | {m['element']}")
            print()

        issues_match = finder.rank("click Issues tab", html, top_k=1)
        if issues_match:
            sel = issues_match[0]["selector"]
            loc = page.locator(f"xpath={sel}").first if sel.startswith("/") else page.locator(sel).first
            await loc.click()
            await page.wait_for_timeout(2000)

        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Folate on GitHub repository pages.")
    parser.add_argument("--headless", action="store_true", help="Run Chromium in headless mode")
    args = parser.parse_args()
    asyncio.run(run_test(headless=args.headless))
