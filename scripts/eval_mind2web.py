import argparse
import gc
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
DEFAULT_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 0.9]


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
    batch_size: int = 32,
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


def score_batch_groups(
    finder: FolateSelectorFinder,
    pending_steps: list[dict],
    batch_size: int = 128,
) -> list[np.ndarray]:
    """Flatten candidate pairs across multiple steps into large ONNX mini-batches for high throughput."""
    flat_pairs = []
    offsets = []
    cursor = 0
    for step in pending_steps:
        pairs = [(step["query"], c["serialized"]) for c in step["pool"]]
        flat_pairs.extend(pairs)
        offsets.append((cursor, cursor + len(pairs)))
        cursor += len(pairs)

    if not flat_pairs:
        return []

    all_logits = []
    for start in range(0, len(flat_pairs), batch_size):
        chunk = flat_pairs[start : start + batch_size]
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

    concatenated = np.concatenate(all_logits, axis=0)
    return [concatenated[s:e] for s, e in offsets]


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
    batch_size: int = 128,
    step_chunk_size: int = 16,
    thresholds: list[float] | None = None,
    confidence_threshold: float = 0.5,
) -> dict:
    thresholds = sorted(set(thresholds or DEFAULT_THRESHOLDS))
    print(f"\nStreaming {dataset_name} split='{split_name}'...", flush=True)
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

    sweep_counts = {
        t: {"tp": 0, "fp": 0, "abstained": 0} for t in thresholds
    }
    cand_tp = 0
    cand_fp = 0
    cand_fn = 0

    # Cache DOM text/parent lookup per unique cleaned_html hash (consecutive steps often share pages)
    last_html_key = None
    last_lookup = {}
    pending_steps = []
    queued_actions = 0

    def flush_pending():
        nonlocal total_actions, top1_hits, top3_hits, top5_hits, mrr_sum, total_ms, total_cands_evaluated, cand_tp, cand_fp, cand_fn
        if not pending_steps:
            return

        t0 = time.perf_counter()
        step_logits_list = score_batch_groups(finder, pending_steps, batch_size=batch_size)
        batch_ms = (time.perf_counter() - t0) * 1000
        per_step_ms = batch_ms / len(pending_steps)

        for step, logits in zip(pending_steps, step_logits_list):
            pool = step["pool"]
            probs = np.clip(logits, 0.0, 1.0)

            ranked_idx = np.argsort(-logits)
            top1_idx = int(ranked_idx[0])
            top1_prob = float(probs[top1_idx])
            top1_is_pos = bool(pool[top1_idx]["is_pos"])

            for t in thresholds:
                if top1_prob >= t:
                    if top1_is_pos:
                        sweep_counts[t]["tp"] += 1
                    else:
                        sweep_counts[t]["fp"] += 1
                else:
                    sweep_counts[t]["abstained"] += 1

            for idx, cand in enumerate(pool):
                pred_pos = float(probs[idx]) >= confidence_threshold
                if pred_pos and cand["is_pos"]:
                    cand_tp += 1
                elif pred_pos and not cand["is_pos"]:
                    cand_fp += 1
                elif not pred_pos and cand["is_pos"]:
                    cand_fn += 1

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

            prev_bucket = total_actions // 100
            total_actions += 1
            total_ms += per_step_ms
            total_cands_evaluated += len(pool)

            if total_actions // 100 > prev_bucket:
                print(
                    f"  [{split_name}] {total_actions} steps | "
                    f"Prec@0: {top1_hits / total_actions * 100:.2f}% | "
                    f"Avg: {total_ms / total_actions:.1f} ms/step",
                    flush=True,
                )
                gc.collect()

        pending_steps.clear()

    for row in ds:
        for act in iter_action_records(row):
            if max_actions is not None and queued_actions >= max_actions:
                break

            pos_cands = act["pos_candidates"]
            neg_cands = act["neg_candidates"]
            if not pos_cands:
                continue

            query = clean_query(act["action_repr"])
            if len(query) < 3:
                continue

            raw_html = act["cleaned_html"]
            html_key = hash(raw_html)
            if html_key != last_html_key:
                last_lookup = build_node_lookup(raw_html)
                last_html_key = html_key
            lookup = last_lookup

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

            pending_steps.append({"query": query, "pool": pool})
            queued_actions += 1

            if len(pending_steps) >= step_chunk_size:
                flush_pending()

        if max_actions is not None and queued_actions >= max_actions:
            break

    flush_pending()

    threshold_metrics = []
    for t in thresholds:
        tp = sweep_counts[t]["tp"]
        fp = sweep_counts[t]["fp"]
        abstained = sweep_counts[t]["abstained"]
        predicted = tp + fp
        prec = (tp / predicted * 100) if predicted else 0.0
        rec = (tp / total_actions * 100) if total_actions else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        cov = (predicted / total_actions * 100) if total_actions else 0.0
        threshold_metrics.append(
            {
                "threshold": t,
                "tp": tp,
                "fp": fp,
                "abstained": abstained,
                "coverage": round(cov, 2),
                "precision": round(prec, 2),
                "recall": round(rec, 2),
                "f1": round(f1, 2),
            }
        )

    active_entry = min(
        threshold_metrics, key=lambda item: abs(item["threshold"] - confidence_threshold)
    )
    cand_prec = (cand_tp / (cand_tp + cand_fp) * 100) if (cand_tp + cand_fp) else 0.0
    cand_rec = (cand_tp / (cand_tp + cand_fn) * 100) if (cand_tp + cand_fn) else 0.0
    cand_f1 = (2 * cand_prec * cand_rec / (cand_prec + cand_rec)) if (cand_prec + cand_rec) else 0.0

    return {
        "split": split_name,
        "steps": total_actions,
        "avg_candidates": round(total_cands_evaluated / total_actions, 1) if total_actions else 0.0,
        "top1_acc": round(top1_hits / total_actions * 100, 2) if total_actions else 0.0,
        "top3_acc": round(top3_hits / total_actions * 100, 2) if total_actions else 0.0,
        "top5_acc": round(top5_hits / total_actions * 100, 2) if total_actions else 0.0,
        "mrr": round(mrr_sum / total_actions, 4) if total_actions else 0.0,
        "confidence_threshold": active_entry["threshold"],
        "precision": active_entry["precision"],
        "recall": active_entry["recall"],
        "f1": active_entry["f1"],
        "coverage": active_entry["coverage"],
        "candidate_precision": round(cand_prec, 2),
        "candidate_recall": round(cand_rec, 2),
        "candidate_f1": round(cand_f1, 2),
        "threshold_sweep": threshold_metrics,
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
                "    - type: precision",
                f"      value: {r['precision']}",
                f"      name: Top-1 Precision @ {r['confidence_threshold']:.2f} ({max_candidates} cands)",
                "    - type: recall",
                f"      value: {r['recall']}",
                f"      name: Top-1 Recall @ {r['confidence_threshold']:.2f} ({max_candidates} cands)",
                "    - type: f1",
                f"      value: {r['f1']}",
                f"      name: Top-1 F1 @ {r['confidence_threshold']:.2f} ({max_candidates} cands)",
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
    parser.add_argument("--batch-size", type=int, default=128, help="ONNX mini-batch size across steps (default: 128)")
    parser.add_argument(
        "--step-chunk-size",
        type=int,
        default=16,
        help="Number of Mind2Web steps to group per batched ONNX forward pass (default: 16)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cuda", "coreml", "cpu"],
        default="auto",
        help="Execution device / ONNX Runtime provider (default: auto)",
    )
    parser.add_argument(
        "--precision-only",
        action="store_true",
        help="Print only Precision / Recall / F1 / Coverage metrics and threshold sweep",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Calibrated [0, 1] confidence threshold for Top-1 Precision / Recall / F1 reporting (default: 0.5)",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Confidence thresholds to sweep for Precision-Recall tradeoff curve",
    )
    parser.add_argument("--model-dir", default=None, help="Optional local model directory override")
    parser.add_argument("--model-id", default="figranium/folate", help="Hugging Face model ID for model-index YAML")
    parser.add_argument("--json-out", default=None, help="Optional path to write JSON evaluation metrics")
    parser.add_argument(
        "--hf-yaml-out",
        default=None,
        help="Optional path to write Hugging Face model-index YAML snippet for README.md metadata",
    )
    args = parser.parse_args()

    provider_map = {
        "auto": None,
        "gpu": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }

    thresholds = sorted(set(list(args.thresholds) + [args.confidence_threshold]))
    finder = FolateSelectorFinder(model_dir=args.model_dir, providers=provider_map[args.device])
    print(f"Active ONNX Runtime providers: {finder.active_providers}", flush=True)

    results = [
        evaluate_split(
            finder=finder,
            dataset_name=args.dataset,
            split_name=split,
            max_actions=args.max_actions,
            max_candidates=args.max_candidates,
            batch_size=args.batch_size,
            step_chunk_size=args.step_chunk_size,
            thresholds=thresholds,
            confidence_threshold=args.confidence_threshold,
        )
        for split in args.splits
    ]

    if args.precision_only:
        print(f"\n{'=' * 92}")
        print("MIND2WEB PRECISION SUMMARY")
        print(f"{'=' * 92}")
        print(
            f"{'Split':<13} | {'Steps':>5} | {'Prec@' + f'{args.confidence_threshold:.2f}':>9} | "
            f"{'Rec@' + f'{args.confidence_threshold:.2f}':>8} | {'F1':>6} | "
            f"{'Coverage':>8} | {'Cand-Prec':>9} | {'Latency':>9}"
        )
        print("-" * 92)
        for r in results:
            print(
                f"{r['split']:<13} | {r['steps']:>5d} | {r['precision']:>8.2f}% | "
                f"{r['recall']:>7.2f}% | {r['f1']:>5.2f}% | {r['coverage']:>7.2f}% | "
                f"{r['candidate_precision']:>8.2f}% | {r['avg_ms_per_step']:>6.1f} ms"
            )
    else:
        print(f"\n{'=' * 106}")
        print("MIND2WEB OFFICIAL TEST SPLIT RESULTS")
        print(f"{'=' * 106}")
        print(
            f"{'Split':<13} | {'Steps':>5} | {'Top-1':>6} | {'Top-3':>6} | {'Top-5':>6} | "
            f"{'MRR':>6} | {'Prec@' + f'{args.confidence_threshold:.2f}':>9} | "
            f"{'Rec@' + f'{args.confidence_threshold:.2f}':>8} | {'F1':>6} | {'Latency':>9}"
        )
        print("-" * 106)
        for r in results:
            print(
                f"{r['split']:<13} | {r['steps']:>5d} | {r['top1_acc']:>5.2f}% | "
                f"{r['top3_acc']:>5.2f}% | {r['top5_acc']:>5.2f}% | {r['mrr']:>6.4f} | "
                f"{r['precision']:>8.2f}% | {r['recall']:>7.2f}% | {r['f1']:>5.2f}% | "
                f"{r['avg_ms_per_step']:>6.1f} ms"
            )

    print(f"\n{'=' * 78}")
    print("PRECISION / RECALL THRESHOLD SWEEP (TOP-1 SELECTIVE PREDICTION)")
    print(f"{'=' * 78}")
    for r in results:
        print(f"\n[{r['split']}] (Candidate-level @ {r['confidence_threshold']:.2f}: P={r['candidate_precision']:.2f}%, R={r['candidate_recall']:.2f}%, F1={r['candidate_f1']:.2f}%)")
        print(f"  {'Thresh':>6} | {'TP':>5} | {'FP':>5} | {'Abstain':>7} | {'Coverage':>8} | {'Precision':>9} | {'Recall':>7} | {'F1':>6}")
        print("  " + "-" * 72)
        for row in r["threshold_sweep"]:
            print(
                f"  {row['threshold']:>6.2f} | {row['tp']:>5d} | {row['fp']:>5d} | "
                f"{row['abstained']:>7d} | {row['coverage']:>7.2f}% | "
                f"{row['precision']:>8.2f}% | {row['recall']:>6.2f}% | {row['f1']:>5.2f}%"
            )

    if not args.precision_only:
        hf_yaml = build_hf_model_index(args.model_id, args.dataset, results, args.max_candidates)
        print(f"\n{'=' * 106}")
        print("HUGGING FACE MODEL CARD METADATA (model-index YAML)")
        print(f"{'=' * 106}")
        print(hf_yaml)

        if args.hf_yaml_out:
            with open(args.hf_yaml_out, "w", encoding="utf-8") as f:
                f.write(hf_yaml)
            print(f"Saved Hugging Face model-index YAML to {args.hf_yaml_out}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset": args.dataset,
                    "providers": finder.active_providers,
                    "max_candidates": args.max_candidates,
                    "confidence_threshold": args.confidence_threshold,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Saved evaluation report to {args.json_out}")


if __name__ == "__main__":
    main()
