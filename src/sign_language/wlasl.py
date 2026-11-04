"""Read and inspect WLASL metadata without downloading any videos."""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path


def load_metadata(path: Path) -> list[dict]:
    """Read the WLASL JSON file and check the fields used by this module."""
    with path.open(encoding="utf-8") as metadata_file:
        signs = json.load(metadata_file)

    if not isinstance(signs, list):
        raise ValueError("WLASL metadata must be a list of gloss entries.")

    for index, sign in enumerate(signs):
        if not isinstance(sign, dict):
            raise ValueError(f"Gloss entry {index} must be an object.")
        if not isinstance(sign.get("gloss"), str) or not sign["gloss"].strip():
            raise ValueError(f"Gloss entry {index} must have a nonempty gloss.")
        if not isinstance(sign.get("instances"), list):
            raise ValueError(f"Gloss entry {index} must have an instances list.")
        for instance in sign["instances"]:
            if not isinstance(instance, dict):
                raise ValueError(f"Each instance in gloss entry {index} must be an object.")
            signer_id = instance.get("signer_id")
            if signer_id is not None and (
                isinstance(signer_id, bool) or not isinstance(signer_id, (int, str))
            ):
                raise ValueError(f"Signer ID in gloss entry {index} must be a number or string.")

    return signs


def find_sign(signs: list[dict], gloss: str) -> dict | None:
    """Find a gloss by name, ignoring case and surrounding input whitespace."""
    requested = gloss.strip().casefold()
    for sign in signs:
        if sign["gloss"].casefold() == requested:
            return sign
    return None


def summarize_sign(sign: dict) -> dict:
    """Count listed clips and distinct known signer IDs for one gloss."""
    instances = sign["instances"]
    signer_ids = {
        instance["signer_id"] for instance in instances if instance.get("signer_id") is not None
    }
    return {"gloss": sign["gloss"], "clips": len(instances), "signers": len(signer_ids)}


def audit_metadata(signs: list[dict]) -> dict:
    """Summarize all listed examples and overlap between their recorded splits."""
    all_signers = set()
    signers_by_split: dict[str, set] = {}
    split_counts: Counter[str] = Counter()
    missing_signer_ids = 0
    missing_urls = 0
    glosses = []

    for sign in signs:
        summary = summarize_sign(sign)
        gloss_split_counts: Counter[str] = Counter()
        for instance in sign["instances"]:
            split = instance.get("split")
            if not isinstance(split, str) or not split.strip():
                split = "(missing)"
            split_counts[split] += 1
            gloss_split_counts[split] += 1

            signer_id = instance.get("signer_id")
            if signer_id is None:
                missing_signer_ids += 1
            else:
                all_signers.add(signer_id)
                signers_by_split.setdefault(split, set()).add(signer_id)

            url = instance.get("url")
            if not isinstance(url, str) or not url.strip():
                missing_urls += 1

        summary["splits"] = dict(sorted(gloss_split_counts.items()))
        glosses.append(summary)

    split_names = sorted(split_counts)
    overlap = {
        f"{left} / {right}": len(
            signers_by_split.get(left, set()) & signers_by_split.get(right, set())
        )
        for left, right in combinations(split_names, 2)
    }
    return {
        "gloss_count": len(signs),
        "listed_clips": sum(split_counts.values()),
        "known_signer_ids": len(all_signers),
        "clips_without_signer_id": missing_signer_ids,
        "clips_without_url": missing_urls,
        "clips_by_source_split": dict(sorted(split_counts.items())),
        "signer_id_overlap_by_split": overlap,
        "glosses": glosses,
    }
