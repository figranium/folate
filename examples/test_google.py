import argparse
import asyncio
import os
import sys
import time
from playwright.async_api import async_playwright

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from folate import FolateSelectorFinder  # noqa: E402


def resolve_locator(page, selector: str):
    if selector.startswith("/"):
        return page.locator(f"xpath={selector}").first
    return page.locator(selector).first


async def run_test(headless: bool = False, query: str = "Figranium browser automation"):
    finder = FolateSelectorFinder()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})

        await page.goto("https://www.google.com", wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)

        html = await page.content()
        candidates = finder.extract_candidates(html)

        t0 = time.perf_counter()
        matches = finder.rank("type search query in Google search box", html, top_k=3)
        ms = (time.perf_counter() - t0) * 1000

        print(f"Ranked {len(candidates)} live Google DOM elements in {ms:.1f} ms:\n")
        for i, m in enumerate(matches, 1):
            print(f"#{i} ({m['score']:.4f}) -> {m['selector']} | {m['element']}")

        if matches:
            locator = resolve_locator(page, matches[0]["selector"])
            await locator.fill(query)
            await page.keyboard.press("Enter")
            await page.wait_for_timeout(2500)

        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Folate on live Google Search.")
    parser.add_argument("--headless", action="store_true", help="Run Chromium in headless mode")
    parser.add_argument("--query", default="Figranium browser automation", help="Search query to enter")
    args = parser.parse_args()
    asyncio.run(run_test(headless=args.headless, query=args.query))
