import argparse
import json
import os
import re
import sys
import time
import numpy as np
from datasets import Image, load_dataset
from lxml import html as lxml_html

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from folate import FolateSelectorFinder  # noqa: E402

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

TEST_SPLITS = ("test_task", "test_website", "test_domain")
DEFAULT_DATASET = "osunlp/Multimodal-Mind2Web"


def parse_candidate_list(raw_cands) -> list[dict]:
    if not raw_cands:
        return []
    if isinstance(raw_cands, str):
        raw_cands = json.loads(raw_cands)
    parsed = []
    for item in raw_cands:
        if isinstance(item, str):
            item = json.loads(item)
        if isinstance(item, dict):
            parsed.append(item)
    return parsed


def clean_query(action_repr: str) -> str:
    desc = action_repr.split("->")[0].strip()
    desc = re.sub(r"[\[\]]", " ", desc)
    return re.sub(r"\s+", " ", desc).strip()


def build_node_lookup(cleaned_html: str) -> dict:
    if not cleaned_html:
        return {}
    tree = lxml_html.fromstring(cleaned_html)
    lookup = {}
    for el in tree.xpath("//*[@backend_node_id]"):
        node_id = el.get("backend_node_id")
        text = " ".join(el.itertext()).strip()
        text = re.sub(r"\s+", " ", text)[:80]
        parent = el.getparent()
        parent_tag = parent.tag if (parent is not None and isinstance(parent.tag, str)) else ""
        lookup[node_id] = {"text": text, "parent": parent_tag}
    return lookup


def serialize_candidate(cand: dict, lookup: dict) -> str:
    tag = cand.get("tag", "").lower()
    raw_attrs = cand.get("attributes", "{}")
    attrs = json.loads(raw_attrs) if isinstance(raw_attrs, str) else raw_attrs

    node_id = str(cand.get("backend_node_id", ""))
    meta = lookup.get(node_id, {"text": "", "parent": ""})

    parts = [f"[TAG] {tag}"]
    for k in KEEP_ATTRS:
        val = attrs.get(k)
        if val:
            parts.append(f"[{k.upper()}] {str(val).strip()[:60]}")

    if meta["text"]:
        parts.append(f"[TEXT] {meta['text']}")
    if meta["parent"]:
        parts.append(f"[PARENT] {meta['parent']}")

    return " ".join(parts)


def score_candidates(
    finder: FolateSelectorFinder,
    query: str,
    candidates: list[dict],
    batch_size: int = 16,
) -> np.ndarray:
    pairs = [(query, c["serialized"]) for c in candidates]
    all_logits = []

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        encodings = finder.tokenizer.encode_batch(chunk)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        logits = finder.session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )[0].reshape(-1)
        all_logits.append(logits)

    return np.concatenate(all_logits, axis=0)


def iter_action_records(row: dict):
    """Yield normalized action dicts from either flat (Multimodal-Mind2Web) or nested (Mind2Web) rows."""
    if "actions" in row and isinstance(row["actions"], list):
        actions = row.get("actions") or []
        action_reprs = row.get("action_reprs") or []
        for i, act in enumerate(actions):
            if i >= len(action_reprs):
                continue
            yield {
                "action_repr": action_reprs[i],
                "cleaned_html": act.get("cleaned_html", ""),
                "pos_candidates": parse_candidate_list(act.get("pos_candidates")),
                "neg_candidates": parse_candidate_list(act.get("neg_candidates")),
            }
        return

    # Flat per-action schema (osunlp/Multimodal-Mind2Web)
    action_repr = row.get("target_action_reprs")
    if not action_repr:
        action_reprs = row.get("action_reprs") or []
        raw_idx = row.get("target_action_index")
        if raw_idx is not None:
            try:
                idx = int(raw_idx)
                if 0 <= idx < len(action_reprs):
                    action_repr = action_reprs[idx]
            except (TypeError, ValueError):
                pass
    if not action_repr:
        return

    yield {
        "action_repr": action_repr,
        "cleaned_html": row.get("cleaned_html", ""),
        "pos_candidates": parse_candidate_list(row.get("pos_candidates")),
        "neg_candidates": parse_candidate_list(row.get("neg_candidates")),
    }


def evaluate_split(
    finder: FolateSelectorFinder,
    dataset_name: str,
    split_name: str,
    max_actions: int | None = None,
    max_candidates: int = 50,
    batch_size: int = 16,
) -> dict:
    print(f"\nStreaming {dataset_name} split='{split_name}'...")
    ds = load_dataset(dataset_name, split=split_name, streaming=True)
    # Disable image decoding on screenshot before removing it so Pillow is never required
    features = getattr(ds, "features", None) or {}
    if "screenshot" in features:
        ds = ds.cast_column("screenshot", Image(decode=False))
        ds = ds.remove_columns(["screenshot"])
    elif getattr(ds, "column_names", None) and "screenshot" in ds.column_names:
        ds = ds.remove_columns(["screenshot"])

    total_actions = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    mrr_sum = 0.0
    total_ms = 0.0
    total_cands_evaluated = 0

    for row in ds:
        for act in iter_action_records(row):
            if max_actions is not None and total_actions >= max_actions:
                break

            pos_cands = act["pos_candidates"]
            neg_cands = act["neg_candidates"]
            if not pos_cands:
                continue

            query = clean_query(act["action_repr"])
            if len(query) < 3:
                continue

            lookup = build_node_lookup(act["cleaned_html"])
            pos_ids = {str(p.get("backend_node_id", "")) for p in pos_cands}

            pool = []
            seen_ids = set()
            for cand in pos_cands + neg_cands:
                nid = str(cand.get("backend_node_id", ""))
                if nid and nid in seen_ids:
                    continue
                if nid:
                    seen_ids.add(nid)
                pool.append(
                    {
                        "node_id": nid,
                        "is_pos": nid in pos_ids,
                        "serialized": serialize_candidate(cand, lookup),
                    }
                )

            if max_candidates and len(pool) > max_candidates:
                positives = [c for c in pool if c["is_pos"]]
                negatives = [c for c in pool if not c["is_pos"]]
                pool = positives + negatives[: max(0, max_candidates - len(positives))]

            t0 = time.perf_counter()
            logits = score_candidates(finder, query, pool, batch_size=batch_size)
            infer_ms = (time.perf_counter() - t0) * 1000

            ranked_idx = np.argsort(-logits)
            first_pos_rank = None
            for rank_pos, idx in enumerate(ranked_idx, start=1):
                if pool[idx]["is_pos"]:
                    first_pos_rank = rank_pos
                    break

            if first_pos_rank is not None:
                if first_pos_rank == 1:
                    top1_hits += 1
                if first_pos_rank <= 3:
                    top3_hits += 1
                if first_pos_rank <= 5:
                    top5_hits += 1
                mrr_sum += 1.0 / first_pos_rank

            total_actions += 1
            total_ms += infer_ms
            total_cands_evaluated += len(pool)

            if total_actions % 100 == 0:
                print(
                    f"  [{split_name}] {total_actions} steps | "
                    f"Top-1: {top1_hits / total_actions * 100:.1f}% | "
                    f"Top-3: {top3_hits / total_actions * 100:.1f}% | "
                    f"Top-5: {top5_hits / total_actions * 100:.1f}% | "
                    f"Avg: {total_ms / total_actions:.1f} ms/step"
                )

        if max_actions is not None and total_actions >= max_actions:
            break

    return {
        "split": split_name,
        "steps": total_actions,
        "avg_candidates": round(total_cands_evaluated / total_actions, 1) if total_actions else 0.0,
        "top1_acc": round(top1_hits / total_actions * 100, 2) if total_actions else 0.0,
        "top3_acc": round(top3_hits / total_actions * 100, 2) if total_actions else 0.0,
        "top5_acc": round(top5_hits / total_actions * 100, 2) if total_actions else 0.0,
        "mrr": round(mrr_sum / total_actions, 4) if total_actions else 0.0,
        "avg_ms_per_step": round(total_ms / total_actions, 2) if total_actions else 0.0,
    }


def build_hf_model_index(model_id: str, dataset_name: str, results: list[dict], max_candidates: int) -> str:
    lines = [
        "model-index:",
        f"- name: {model_id}",
        "  results:",
    ]
    for r in results:
        split_label = r["split"].replace("_", " ").title()
        lines.extend(
            [
                "  - task:",
                "      type: text-classification",
                "      name: DOM Element Grounding",
                "    dataset:",
                f"      type: {dataset_name}",
                f"      name: Mind2Web ({split_label})",
                f"      split: {r['split']}",
                "    metrics:",
                "    - type: accuracy",
                f"      value: {r['top1_acc']}",
                f"      name: Top-1 Accuracy ({max_candidates} cands)",
                "    - type: recall",
                f"      value: {r['top3_acc']}",
                f"      name: Top-3 Recall ({max_candidates} cands)",
                "    - type: recall",
                f"      value: {r['top5_acc']}",
                f"      name: Top-5 Recall ({max_candidates} cands)",
                "    - type: mrr",
                f"      value: {r['mrr']}",
                f"      name: Mean Reciprocal Rank ({max_candidates} cands)",
            ]
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Folate on official Mind2Web test splits.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset containing Mind2Web test splits (default: osunlp/Multimodal-Mind2Web)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(TEST_SPLITS),
        choices=list(TEST_SPLITS),
        help="Mind2Web test splits to evaluate (default: test_task test_website test_domain)",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="Optional cap on evaluated actions per split (omit for full official test split)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help="Candidate pool size per action (default: 50, matching MindAct stage-1 protocol; 0 = all)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="ONNX mini-batch size")
    parser.add_argument("--model-dir", default=None, help="Optional local model directory override")
    parser.add_argument("--model-id", default="figranium/folate", help="Hugging Face model ID for model-index YAML")
    parser.add_argument("--json-out", default=None, help="Optional path to write JSON evaluation metrics")
    parser.add_argument(
        "--hf-yaml-out",
        default=None,
        help="Optional path to write Hugging Face model-index YAML snippet for README.md metadata",
    )
    args = parser.parse_args()

    finder = FolateSelectorFinder(model_dir=args.model_dir)
    results = [
        evaluate_split(
            finder=finder,
            dataset_name=args.dataset,
            split_name=split,
            max_actions=args.max_actions,
            max_candidates=args.max_candidates,
            batch_size=args.batch_size,
        )
        for split in args.splits
    ]

    print(f"\n{'=' * 82}")
    print("MIND2WEB OFFICIAL TEST SPLIT RESULTS")
    print(f"{'=' * 82}")
    print(f"{'Split':<15} | {'Steps':>6} | {'Cands':>5} | {'Top-1':>7} | {'Top-3':>7} | {'Top-5':>7} | {'MRR':>6} | {'Latency':>9}")
    print("-" * 82)
    for r in results:
        print(
            f"{r['split']:<15} | {r['steps']:>6d} | {r['avg_candidates']:>5.1f} | "
            f"{r['top1_acc']:>6.2f}% | {r['top3_acc']:>6.2f}% | {r['top5_acc']:>6.2f}% | "
            f"{r['mrr']:>6.4f} | {r['avg_ms_per_step']:>6.1f} ms"
        )

    hf_yaml = build_hf_model_index(args.model_id, args.dataset, results, args.max_candidates)
    print(f"\n{'=' * 82}")
    print("HUGGING FACE MODEL CARD METADATA (model-index YAML)")
    print(f"{'=' * 82}")
    print(hf_yaml)

    if args.hf_yaml_out:
        with open(args.hf_yaml_out, "w", encoding="utf-8") as f:
            f.write(hf_yaml)
        print(f"Saved Hugging Face model-index YAML to {args.hf_yaml_out}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"dataset": args.dataset, "max_candidates": args.max_candidates, "results": results}, f, indent=2)
        print(f"Saved evaluation report to {args.json_out}")


if __name__ == "__main__":
    main()
