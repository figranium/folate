import os
import re
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from lxml import html as lxml_html
from tokenizers import Tokenizer

DEFAULT_REPO_ID = "figranium/folate"

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

STOP_WORDS = {
    "a", "an", "the", "to", "in", "on", "at", "for", "of", "with", "by",
    "click", "type", "enter", "fill", "select", "open", "press", "tap",
    "button", "link", "tab", "box", "field", "input", "dropdown", "icon",
}


class FolateSelectorFinder:
    """Lightweight cross-encoder ranker for webpage DOM elements (CPU & GPU)."""

    def __init__(
        self,
        model_dir: str | None = None,
        repo_id: str = DEFAULT_REPO_ID,
        revision: str | None = None,
        num_threads: int = 4,
        providers: list[str] | None = None,
        model_filename: str | None = None,
    ):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = ort.get_available_providers()
        if providers is None:
            providers = [
                p
                for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
                if p in available
            ] or ["CPUExecutionProvider"]

        # Note: INT8 QUInt8 dynamic models run fastest on CPU; FP32 ONNX runs natively on CUDA/CoreML.
        if model_filename is None:
            using_gpu = any(p != "CPUExecutionProvider" for p in providers)
            model_filename = "model.onnx" if using_gpu else "model_quantized.onnx"

        base_dir = None
        if model_dir is not None and os.path.isdir(model_dir):
            base_dir = model_dir
        elif os.path.isdir("./folate-int8"):
            base_dir = "./folate-int8"

        if base_dir is not None and os.path.isfile(os.path.join(base_dir, model_filename)):
            model_path = os.path.join(base_dir, model_filename)
            tokenizer_path = os.path.join(base_dir, "tokenizer.json")
        elif (
            base_dir is not None
            and model_filename != "model_quantized.onnx"
            and os.path.isfile(os.path.join(base_dir, "model_quantized.onnx"))
        ):
            model_path = os.path.join(base_dir, "model_quantized.onnx")
            tokenizer_path = os.path.join(base_dir, "tokenizer.json")
        else:
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=model_filename,
                    revision=revision,
                )
            except Exception:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename="model_quantized.onnx",
                    revision=revision,
                )
            tokenizer_path = hf_hub_download(
                repo_id=repo_id,
                filename="tokenizer.json",
                revision=revision,
            )

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers,
        )
        self.active_providers = self.session.get_providers()
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

    @staticmethod
    def _prefilter_candidates(
        query: str, candidates: list[dict], max_candidates: int
    ) -> list[dict]:
        if len(candidates) <= max_candidates:
            return candidates

        terms = [
            t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in STOP_WORDS
        ]
        if not terms:
            return candidates[:max_candidates]

        scored = []
        for idx, cand in enumerate(candidates):
            haystack = cand["serialized"].lower()
            hits = sum(1 for t in terms if t in haystack)
            scored.append((hits, -idx, cand))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in scored[:max_candidates]]

    def rank(
        self,
        query: str,
        page_html: str,
        top_k: int = 3,
        max_candidates: int = 48,
        batch_size: int = 16,
    ) -> list[dict]:
        candidates = self.extract_candidates(page_html)
        if not candidates:
            return []

        candidates = self._prefilter_candidates(query, candidates, max_candidates)
        pairs = [(query, c["serialized"]) for c in candidates]
        all_logits = []

        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            encodings = self.tokenizer.encode_batch(chunk)

            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            )
            token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

            logits = self.session.run(
                ["logits"],
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )[0].reshape(-1)
            all_logits.append(logits)

        logits = np.concatenate(all_logits, axis=0)
        scores = np.clip(logits, 0.0, 1.0)
        ranked_indices = np.argsort(-logits)[:top_k]

        return [
            {
                "score": float(scores[idx]),
                "logit": float(logits[idx]),
                "selector": candidates[idx]["selector"],
                "element": candidates[idx]["serialized"],
            }
            for idx in ranked_indices
        ]
