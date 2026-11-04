"""Create reproducible, provisional WLASL splits from metadata."""

import random
from collections import Counter, defaultdict

SPLITS = ("train", "val", "test")


def split_metadata(signs: list[dict], labels: tuple[str, ...], seed: int = 42) -> dict:
    """Assign signer and shared-media groups without asserting video availability."""
    by_gloss = {}
    for sign in signs:
        key = sign["gloss"].casefold()
        if key in by_gloss:
            raise ValueError(f"Duplicate gloss in metadata: {sign['gloss']}")
        by_gloss[key] = sign

    records = []
    for class_id, label in enumerate(labels):
        sign = by_gloss.get(label.casefold())
        if sign is None:
            raise ValueError(f"Vocabulary gloss not found in metadata: {label}")
        if not sign["instances"]:
            raise ValueError(f"Vocabulary gloss has no listed clips: {label}")
        for instance in sign["instances"]:
            signer = instance.get("signer_id")
            video_id = instance.get("video_id")
            url = instance.get("url")
            if signer is None or (isinstance(signer, str) and not signer.strip()):
                raise ValueError(f"Missing signer ID for gloss: {label}")
            if not isinstance(video_id, str) or not video_id.strip():
                raise ValueError(f"Missing video ID for gloss: {label}")
            if not isinstance(url, str) or not url.strip():
                raise ValueError(f"Missing URL for video: {video_id}")
            records.append(
                {
                    "class_id": class_id,
                    "gloss": sign["gloss"],
                    "video_id": video_id,
                    "signer_id": signer,
                    "source": instance.get("source"),
                    "url": url,
                    "source_split": instance.get("split"),
                    "fps": instance.get("fps"),
                    "frame_start": instance.get("frame_start"),
                    "frame_end": instance.get("frame_end"),
                    "content_sha256": instance.get("content_sha256"),
                    "availability": "not_checked",
                }
            )

    signer_types = {type(record["signer_id"]) for record in records}
    if len(signer_types) > 1:
        raise ValueError("Mixed signer ID types must be resolved before splitting.")

    # A repeated signer or exact media identifier must stay in a single group.
    parents = {record["signer_id"]: record["signer_id"] for record in records}

    def find(signer):
        while parents[signer] != signer:
            parents[signer] = parents[parents[signer]]
            signer = parents[signer]
        return signer

    owners = {}
    for record in records:
        signer = record["signer_id"]
        keys = [("video_id", record["video_id"]), ("url", record["url"])]
        if record["content_sha256"]:
            keys.append(("sha256", record["content_sha256"]))
        for key in keys:
            if key in owners:
                parents[find(signer)] = find(owners[key])
            else:
                owners[key] = signer

    groups = {}
    for signer in sorted(parents, key=lambda value: (type(value).__name__, str(value))):
        groups.setdefault(find(signer), []).append(signer)
    shuffled_groups = list(groups.values())
    if len(shuffled_groups) < 3:
        raise ValueError("At least three independent signer/media groups are required.")
    random.Random(seed).shuffle(shuffled_groups)
    total = len(shuffled_groups)
    test_count = max(1, round(total * 0.2))
    val_count = max(1, round(total * 0.2))
    train_end = total - test_count - val_count
    if train_end < 1:
        raise ValueError("Not enough groups to populate all three splits.")
    assigned_groups = {
        "train": shuffled_groups[:train_end],
        "val": shuffled_groups[train_end : train_end + val_count],
        "test": shuffled_groups[train_end + val_count :],
    }
    assignments = {
        signer: split
        for split, split_groups in assigned_groups.items()
        for group in split_groups
        for signer in group
    }
    for record in records:
        record["assigned_split"] = assignments[record["signer_id"]]

    counts = Counter(record["assigned_split"] for record in records)
    records_by_class = defaultdict(list)
    for record in records:
        records_by_class[record["class_id"]].append(record)
    coverage = []
    for class_id, label in enumerate(labels):
        class_records = records_by_class[class_id]
        by_split = {}
        for split in SPLITS:
            items = [record for record in class_records if record["assigned_split"] == split]
            by_split[split] = {
                "clips": len(items),
                "signer_ids": len({item["signer_id"] for item in items}),
            }
        coverage.append({"class_id": class_id, "gloss": label, "splits": by_split})

    missing = {
        split: [item["gloss"] for item in coverage if item["splits"][split]["clips"] == 0]
        for split in SPLITS
    }
    signer_sets = {
        split: {record["signer_id"] for record in records if record["assigned_split"] == split}
        for split in SPLITS
    }
    url_sets = {
        split: {record["url"] for record in records if record["assigned_split"] == split}
        for split in SPLITS
    }
    video_sets = {
        split: {record["video_id"] for record in records if record["assigned_split"] == split}
        for split in SPLITS
    }
    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{left} / {right}"] = {
            "signer_ids": len(signer_sets[left] & signer_sets[right]),
            "exact_urls": len(url_sets[left] & url_sets[right]),
            "video_ids": len(video_sets[left] & video_sets[right]),
        }

    return {
        "status": "provisional_metadata_only",
        "method": "seeded signer groups joined by exact URL, video ID or available content hash",
        "seed": seed,
        "label_order": list(labels),
        "group_counts": {split: len(assigned_groups[split]) for split in SPLITS},
        "clip_counts": {split: counts[split] for split in SPLITS},
        "signer_counts": {split: len(signer_sets[split]) for split in SPLITS},
        "overlap": overlaps,
        "missing_glosses": missing,
        "coverage": coverage,
        "records": records,
    }


def split_with_coverage(
    signs: list[dict], labels: tuple[str, ...], seed: int = 42, attempts: int = 256
) -> dict:
    """Choose a deterministic signer-disjoint split using labels, not model results."""
    if attempts < 1:
        raise ValueError("At least one split attempt is required.")
    best = None
    best_score = None
    for offset in range(attempts):
        candidate = split_metadata(signs, labels, seed + offset)
        missing = candidate["missing_glosses"]
        counts = candidate["clip_counts"]
        total = sum(counts.values())
        score = (
            len(missing["train"]),
            len(missing["val"]) + len(missing["test"]),
            abs(counts["train"] / total - 0.6),
            abs(counts["val"] - counts["test"]),
        )
        if best_score is None or score < best_score:
            best, best_score = candidate, score
    best["selection"] = {
        "method": "minimize missing train classes, then missing validation/test classes",
        "initial_seed": seed,
        "attempts": attempts,
        "selected_seed": best["seed"],
    }
    return best
