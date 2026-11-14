# Folate

Lightweight, CPU-efficient neural DOM element ranker built for browser automation within tight memory constraints (<2 GB total footprint alongside a headless browser).

Folate fine-tunes [`google/electra-base-discriminator`](https://huggingface.co/google/electra-base-discriminator) as a cross-encoder ranker on `osunlp/Mind2Web` and exports to an **INT8-quantized ONNX model (~107 MB)** that runs purely via `onnxruntime` and `tokenizers`—requiring no PyTorch or `transformers` at inference time.

## Architecture & Footprint

- **Backbone:** `google/electra-base-discriminator` (cross-encoder, single logit output)
- **Max Sequence Length:** `96` tokens
- **Serialized Element Schema:** `[TAG] ... [ID/NAME/ROLE/ARIA-LABEL/PLACEHOLDER/HREF/DATA-TESTID] ... [TEXT] ... [PARENT] ...`
- **Quantized Model Size:** ~107 MB (`QUInt8` dynamic quantization)
- **Runtime RAM:** ~180–250 MB (CPU-only `onnxruntime` + `tokenizers` + `lxml`)

## Project Structure

```text
folate/
├── folate/
│   ├── __init__.py
│   └── ranker.py          # Lightweight CPU inference engine & DOM serializer
├── scripts/
│   ├── prepare_data.py    # Streaming Mind2Web serializer (lxml)
│   ├── train.py           # Fine-tuning script (bf16/fp16/MPS/CPU)
│   └── export_onnx.py     # FP32 ONNX export + INT8 dynamic quantization
├── examples/
│   └── benchmark.py       # Local latency, RAM, and accuracy sanity benchmark
├── requirements.txt       # Lightweight inference dependencies
├── requirements-train.txt # Training & ONNX export dependencies
└── .gitignore
```

## Quickstart (Inference)

1. **Install lightweight runtime dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Place your quantized model artifacts** (`model_quantized.onnx`, `tokenizer.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`) inside `./folate-int8/`.

3. **Find DOM elements with natural language:**
   ```python
   from folate import FolateSelectorFinder

   finder = FolateSelectorFinder(model_dir="./folate-int8")
   matches = finder.rank("click Search Flights button", html_source, top_k=3)

   for match in matches:
       print(f"{match['score']:.4f} | {match['selector']} | {match['element']}")
   ```

4. **Run the local benchmark:**
   ```bash
   python examples/benchmark.py
   ```

## Training & Quantization Pipeline

To reproduce training from scratch on `osunlp/Mind2Web`:

```bash
pip install -r requirements-train.txt

# 1. Stream and serialize Mind2Web pairs (~1-2 mins)
python scripts/prepare_data.py --output-dir ./data

# 2. Fine-tune ELECTRA cross-encoder
python scripts/train.py --data-dir ./data --save-dir ./folate-base

# 3. Export to FP32 ONNX and quantize to INT8 (~107 MB)
python scripts/export_onnx.py --model-dir ./folate-base --output-dir ./folate-int8
```

## License

Licensed under the terms in [LICENSE](LICENSE).
