import argparse
import json
import os
import random
import re
from datasets import load_dataset
from lxml import html as lxml_html

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


def main():
    parser = argparse.ArgumentParser(description="Stream and serialize Mind2Web for Folate.")
    parser.add_argument("--output-dir", default="./data", help="Output directory for train.jsonl and val.jsonl")
    parser.add_argument("--negatives-per-pos", type=int, default=7, help="Number of negative samples per positive")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    print("Streaming osunlp/Mind2Web...")
    ds = load_dataset("osunlp/Mind2Web", split="train", streaming=True)

    grouped_examples = []
    for task_idx, task in enumerate(ds, start=1):
        actions = task.get("actions") or []
        action_reprs = task.get("action_reprs") or []

        for i, act in enumerate(actions):
            pos_cands = act.get("pos_candidates") or []
            neg_cands = act.get("neg_candidates") or []
            if not pos_cands or not neg_cands or i >= len(action_reprs):
                continue

            query = clean_query(action_reprs[i])
            if len(query) < 3:
                continue

            lookup = build_node_lookup(act.get("cleaned_html", ""))
            pos_text = serialize_candidate(pos_cands[0], lookup)
            sampled_negs = rng.sample(neg_cands, min(args.negatives_per_pos, len(neg_cands)))
            neg_texts = [serialize_candidate(n, lookup) for n in sampled_negs]

            grouped_examples.append(
                {
                    "query": query,
                    "positive": pos_text,
                    "negatives": neg_texts,
                }
            )

        if task_idx % 100 == 0:
            print(f"Processed {task_idx} tasks ({len(grouped_examples)} examples collected)...")

    rng.shuffle(grouped_examples)
    split_idx = int(len(grouped_examples) * 0.9)

    splits = {
        "train.jsonl": grouped_examples[:split_idx],
        "val.jsonl": grouped_examples[split_idx:],
    }

    for filename, subset in splits.items():
        out_path = os.path.join(args.output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            for group in subset:
                f.write(json.dumps({"query": group["query"], "element": group["positive"], "label": 1}) + "\n")
                for neg in group["negatives"]:
                    f.write(json.dumps({"query": group["query"], "element": neg, "label": 0}) + "\n")

    print(f"Done! Created {split_idx} train groups and {len(grouped_examples) - split_idx} val groups in {args.output_dir}.")


if __name__ == "__main__":
    main()
