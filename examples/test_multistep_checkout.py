import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from folate import FolateSelectorFinder  # noqa: E402

CHECKOUT_HTML = """
<html><body>
  <main class="checkout-container">
    <section aria-label="Product Options">
      <select id="size-select" name="size" aria-label="Select shoe size">
        <option value="9">US 9</option>
        <option value="10">US 10</option>
      </select>
      <button id="add-to-bag" data-testid="add-to-cart-btn">Add to Bag</button>
    </section>

    <form id="shipping-form">
      <input type="text" id="full-name" name="fullName" placeholder="Full name" />
      <input type="email" id="email-addr" name="email" placeholder="Email address for receipt" />
      <input type="text" id="street-address" name="address1" placeholder="Street address" />
      <input type="text" id="postal-code" name="zip" placeholder="ZIP / Postal code" />
      <input type="checkbox" id="billing-same" name="sameAsShipping" aria-label="Billing address same as shipping" />
    </form>

    <section aria-label="Discount and Payment">
      <input type="text" id="promo-input" name="promoCode" placeholder="Enter promo or gift code" />
      <button type="button" id="apply-promo">Apply</button>
      <button type="submit" id="place-order-btn" data-testid="submit-order">Place Order</button>
    </section>
  </main>
</body></html>
"""

STEPS = [
    ("select shoe size dropdown", "select#size-select"),
    ("click Add to Bag button", "#add-to-bag"),
    ("type full name in shipping form", "#full-name"),
    ("enter email address", "#email-addr"),
    ("fill street address field", "#street-address"),
    ("enter postal code", "#postal-code"),
    ("check billing address same as shipping", "#billing-same"),
    ("enter promo code", "#promo-input"),
    ("click Apply promo button", "#apply-promo"),
    ("click Place Order button", "#place-order-btn"),
]


def main():
    finder = FolateSelectorFinder()
    total_ms = 0.0
    hits = 0

    for step_idx, (query, expected_selector) in enumerate(STEPS, 1):
        t0 = time.perf_counter()
        matches = finder.rank(query, CHECKOUT_HTML, top_k=1)
        ms = (time.perf_counter() - t0) * 1000
        total_ms += ms

        top = matches[0]
        matched = expected_selector in top["selector"] or top["selector"] in expected_selector
        hits += int(matched)
        status = "PASS" if matched else "MISS"
        print(f"[{status}] Step {step_idx:02d} ({ms:5.1f} ms) | Query: '{query}'")
        print(f"       Predicted: {top['selector']} (score={top['score']:.4f}) | Expected: {expected_selector}")

    print(f"\nTop-1 Accuracy: {hits}/{len(STEPS)} ({hits / len(STEPS) * 100:.0f}%) | Total CPU Time: {total_ms:.1f} ms")


if __name__ == "__main__":
    main()
