import os
import re
import numpy as np
import onnxruntime as ort
from lxml import html as lxml_html
from tokenizers import Tokenizer

KEEP_ATTRS = [
    "id",
    "name",
    "type",
    "role",
    "aria-label",
    "placeholder",
    "title",
    "alt",
    "href",
    "data-testid",
]

INTERACTIVE_XPATH = (
    "//a | //button | //input | //select | //textarea | "
    "//*[@role='button' or @role='link' or @role='menuitem' or @role='tab' or @role='checkbox' or @role='combobox'] | "
    "//*[@onclick or @tabindex]"
)


class FolateSelectorFinder:
    """Lightweight CPU cross-encoder ranker for webpage DOM elements."""

    def __init__(self, model_dir: str = "./folate-int8", num_threads: int = 4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        model_path = os.path.join(model_dir, "model_quantized.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=96)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    @staticmethod
    def serialize_element(el) -> str:
        tag = el.tag.lower() if isinstance(el.tag, str) else ""
        parts = [f"[TAG] {tag}"]

        for attr in KEEP_ATTRS:
            val = el.get(attr)
            if val:
                parts.append(f"[{attr.upper()}] {str(val).strip()[:60]}")

        text = " ".join(el.itertext()).strip()
        text = re.sub(r"\s+", " ", text)[:80]
        if text:
            parts.append(f"[TEXT] {text}")

        parent = el.getparent()
        if parent is not None and isinstance(parent.tag, str):
            parts.append(f"[PARENT] {parent.tag.lower()}")

        return " ".join(parts)

    @staticmethod
    def build_css_selector(el) -> str:
        tag = el.tag.lower()
        el_id = el.get("id")
        if el_id and re.match(r"^[A-Za-z_][\w-]*$", el_id):
            return f"#{el_id}"
        for attr in ("data-testid", "name", "aria-label", "placeholder", "href"):
            val = el.get(attr)
            if val and '"' not in val and len(val) < 80:
                return f'{tag}[{attr}="{val}"]'
        return el.getroottree().getpath(el)

    def extract_candidates(self, page_html: str) -> list[dict]:
        if not page_html:
            return []
        tree = lxml_html.fromstring(page_html)
        seen = set()
        candidates = []

        for el in tree.xpath(INTERACTIVE_XPATH):
            if not isinstance(el.tag, str):
                continue
            if el.tag.lower() == "input" and (el.get("type") or "").lower() == "hidden":
                continue

            serialized = self.serialize_element(el)
            selector = self.build_css_selector(el)
            key = (selector, serialized)
            if key in seen:
                continue
            seen.add(key)

            candidates.append(
                {
                    "selector": selector,
                    "serialized": serialized,
                }
            )
        return candidates

    def rank(self, query: str, page_html: str, top_k: int = 3) -> list[dict]:
        candidates = self.extract_candidates(page_html)
        if not candidates:
            return []

        pairs = [(query, c["serialized"]) for c in candidates]
        encodings = self.tokenizer.encode_batch(pairs)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        logits = self.session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )[0].reshape(-1)

        probs = 1.0 / (1.0 + np.exp(-logits))
        ranked_indices = np.argsort(-probs)[:top_k]

        return [
            {
                "score": float(probs[idx]),
                "selector": candidates[idx]["selector"],
                "element": candidates[idx]["serialized"],
            }
            for idx in ranked_indices
        ]
