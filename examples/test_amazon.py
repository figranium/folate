import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from folate import FolateSelectorFinder  # noqa: E402

FALLBACK_AMAZON_HTML = """
<html><body>
  <header id="navbar">
    <a id="nav-logo-sprites" href="/" aria-label="Amazon">Amazon</a>
    <form id="nav-search-bar-form" role="search">
      <select id="searchDropdownBox" aria-label="Search in">
        <option value="all">All Departments</option>
        <option value="electronics">Electronics</option>
      </select>
      <input type="text" id="twotabsearchtextbox" name="field-keywords" placeholder="Search Amazon" aria-label="Search Amazon" />
      <input type="submit" id="nav-search-submit-button" value="Go" aria-label="Go" />
    </form>
    <a id="nav-link-accountList" href="/ap/signin">Hello, sign in Account & Lists</a>
    <a id="nav-orders" href="/gp/css/order-history">Returns & Orders</a>
    <a id="nav-cart" href="/gp/cart/view.html" aria-label="0 items in shopping cart">Cart</a>
  </header>
  <nav id="nav-main">
    <a href="/gp/bestsellers">Best Sellers</a>
    <a href="/deals">Today's Deals</a>
    <a href="/prime">Prime</a>
    <a href="/gift-cards">Gift Cards</a>
  </nav>
  <main>
    <div data-component-type="s-search-result">
      <h2><a href="/dp/B09G9FPHY6">Apple MacBook Air M2 Laptop</a></h2>
      <button id="a-autoid-1-announce" aria-label="Add Apple MacBook Air M2 Laptop to cart">Add to cart</button>
    </div>
  </main>
</body></html>
"""

QUERIES = [
    "type product name in Amazon search box",
    "click search submit button",
    "open shopping cart",
    "click Today's Deals link",
    "click Add to cart button",
]


def evaluate_html(finder: FolateSelectorFinder, html: str, source_label: str):
    candidates = finder.extract_candidates(html)
    print(f"Source: {source_label} ({len(candidates)} interactive elements extracted)\n")

    for query in QUERIES:
        t0 = time.perf_counter()
        matches = finder.rank(query, html, top_k=2)
        ms = (time.perf_counter() - t0) * 1000
        print(f"Query: '{query}' ({ms:.1f} ms)")
        for rank_i, m in enumerate(matches, 1):
            print(f"  #{rank_i} ({m['score']:.4f}) -> {m['selector']} | {m['element']}")
        print()


async def run_live(headless: bool = False):
    from playwright.async_api import async_playwright

    finder = FolateSelectorFinder()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page(viewport={"width": 1366, "height": 900})

        try:
            await page.goto("https://www.amazon.com", wait_until="domcontentloaded", timeout=15000)
            await page.wait_for_timeout(2000)
            html = await page.content()
            if "twotabsearchtextbox" in html:
                evaluate_html(finder, html, "Live https://www.amazon.com")
                search_match = finder.rank("type product name in Amazon search box", html, top_k=1)
                if search_match:
                    sel = search_match[0]["selector"]
                    loc = page.locator(f"xpath={sel}").first if sel.startswith("/") else page.locator(sel).first
                    await loc.fill("wireless mechanical keyboard")
                    await page.keyboard.press("Enter")
                    await page.wait_for_timeout(2500)
            else:
                print("Amazon returned a bot-check interstitial; running against realistic Amazon DOM snapshot.\n")
                evaluate_html(finder, FALLBACK_AMAZON_HTML, "Fallback Amazon DOM Snapshot")
        except Exception as exc:
            print(f"Live navigation failed ({exc}); running against realistic Amazon DOM snapshot.\n")
            evaluate_html(finder, FALLBACK_AMAZON_HTML, "Fallback Amazon DOM Snapshot")
        finally:
            await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Folate element ranking on Amazon.")
    parser.add_argument("--headless", action="store_true", help="Run Chromium in headless mode")
    parser.add_argument("--offline", action="store_true", help="Run without launching a browser using the fallback DOM")
    args = parser.parse_args()

    if args.offline:
        evaluate_html(FolateSelectorFinder(), FALLBACK_AMAZON_HTML, "Offline Amazon DOM Snapshot")
    else:
        asyncio.run(run_live(headless=args.headless))
