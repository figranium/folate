<p align="center">
  <img src="banner.png" alt="Folate" width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/figranium/folate"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg" alt="Model on Hugging Face"></a>
  &nbsp;&bull;&nbsp;
  <a href="https://github.com/figranium/folate"><strong>GitHub</strong></a>
  &nbsp;&bull;&nbsp;
  <a href="LICENSE"><strong>Apache 2.0 License</strong></a>
</p>

Folate is a lightweight, CPU-efficient neural DOM element ranker built for browser automation within tight memory constraints (<2 GB total footprint alongside a headless browser). Fine-tuned from [`google/electra-base-discriminator`](https://huggingface.co/google/electra-base-discriminator) on [`osunlp/Mind2Web`](https://huggingface.co/datasets/osunlp/Mind2Web) and dynamically quantized to **INT8 ONNX (~107 MB)**, Folate runs purely via `onnxruntime` and `tokenizers`—requiring no PyTorch or `transformers` at inference time.

---

## Model Overview

Folate operates as a discriminative cross-encoder that scores `(natural_language_query, serialized_dom_element)` pairs in a single batched forward pass on CPU.

| Variant / Artifact | Backbone | Precision / Format | Disk Size | Runtime RAM |
| :--- | :--- | :--- | :--- | :--- |
| **`folate-int8` (Default)** | `google/electra-base-discriminator` | Dynamic `QUInt8` ONNX | ~107 MB | ~180–250 MB |
| **`folate-base`** | `google/electra-base-discriminator` | `float32` Safetensors | ~418 MB | ~1.4+ GB (PyTorch) |

### Key Capabilities

* **Zero-Hallucination Selector Grounding** — Ranks real interactive DOM nodes extracted via `lxml` XPath (`<a>`, `<button>`, `<input>`, `<select>`, `<textarea>`, ARIA interactive roles, and `onclick`/`tabindex` elements), guaranteeing every returned CSS/XPath selector exists on the live page.
* **Compact Structured Serialization** — Serializes each element into a concise `96`-token representation:
  `[TAG] ... [ID/NAME/TYPE/ROLE/ARIA-LABEL/PLACEHOLDER/TITLE/ALT/HREF/DATA-TESTID] ... [TEXT] ... [PARENT] ...`
* **Edge & Low-Memory Deployment** — Designed to run alongside Chromium on resource-constrained laptops and containers without GPU acceleration.

---

## Quickstart

### Installation

```bash
git clone https://github.com/figranium/folate
cd folate
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To run the live browser examples with Playwright:

```bash
python -m pip install -r examples/requirements.txt
python -m playwright install chromium
```

### Usage

`FolateSelectorFinder` automatically downloads and caches [`figranium/folate`](https://huggingface.co/figranium/folate) on first run (or loads `./folate-int8` if present locally):

```python
from folate import FolateSelectorFinder

finder = FolateSelectorFinder()
matches = finder.rank("click Search Flights button", html_source, top_k=3)

for match in matches:
    print(f"{match['score']:.4f} | {match['selector']} | {match['element']}")
```

To load weights from a custom local directory:

```python
finder = FolateSelectorFinder(model_dir="./folate-int8")
```

### Programmatic Model Download

If you want to pre-fetch the quantized model weights via `huggingface_hub`:

```bash
hf download figranium/folate --local-dir ./folate-int8
```

---

## Repository Structure

```text
folate/
├── folate/
│   ├── __init__.py
│   └── ranker.py                  # Lightweight CPU inference engine & DOM serializer
├── scripts/
│   ├── prepare_data.py            # Streaming Mind2Web serializer (lxml)
│   ├── train.py                   # Fine-tuning script (bf16/fp16/CPU)
│   └── export_onnx.py             # FP32 ONNX export + INT8 dynamic quantization
├── examples/
│   ├── benchmark.py               # Latency, RSS memory, and top-k sanity benchmark
│   ├── test_google.py             # Live Playwright search box grounding on google.com
│   ├── test_amazon.py             # E-commerce element ranking on amazon.com (+ offline snapshot)
│   ├── test_github.py             # Repository navigation grounding on github.com
│   ├── test_multistep_checkout.py # 10-step e-commerce checkout flow benchmark
│   └── requirements.txt           # Optional Playwright dependency for live browser tests
├── requirements.txt               # Lightweight CPU runtime dependencies
├── requirements-train.txt         # Training & ONNX export dependencies
└── LICENSE
```

---

## Examples & Benchmarks

Run any of the included benchmarks or live browser tests from the repository root:

```bash
python examples/benchmark.py
python examples/test_multistep_checkout.py
python examples/test_google.py
python examples/test_amazon.py --offline
python examples/test_github.py
```

---

## Training & Quantization Pipeline

To reproduce data preparation, fine-tuning, and INT8 ONNX export from scratch on [`osunlp/Mind2Web`](https://huggingface.co/datasets/osunlp/Mind2Web):

```bash
python -m pip install -r requirements-train.txt

# 1. Stream and serialize Mind2Web pairs
python scripts/prepare_data.py --output-dir ./data

# 2. Fine-tune ELECTRA cross-encoder
python scripts/train.py --data-dir ./data --save-dir ./folate-base

# 3. Export to FP32 ONNX and quantize to INT8 (~107 MB)
python scripts/export_onnx.py --model-dir ./folate-base --output-dir ./folate-int8
```

---

## Intended Use & Limitations

* **Intended Use:** Single-step natural-language UI element grounding (`click`, `type`, `select`, `hover`) in local browser automation pipelines where RAM and CPU budgets are constrained.
* **Limitations:**
  * **No Multi-Step Planning:** Folate ranks candidate elements for a single action query; it does not plan multi-step workflows or parse unstructured page content.
  * **Local DOM Context Only:** Elements are serialized using their own attributes, inner text, and immediate `[PARENT]` tag—identical sibling elements without distinguishing attributes or text cannot be disambiguated visually.

---

## License

Folate is released under the [Apache 2.0 License](LICENSE).
