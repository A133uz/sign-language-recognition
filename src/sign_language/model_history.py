"""Read preserved model metadata after checkpoint curation; never restore weights."""
from pathlib import Path

HISTORY='data/model-history/2026-11-14'


def artifact_path(root,relative,*,prefer_history=False):
    root=Path(root).resolve();relative=Path(relative)
    if relative.is_absolute():relative=relative.resolve().relative_to(root)
    if '..' in relative.parts:raise ValueError('Artifact path leaves workspace')
    active=root/relative
    historical=root/HISTORY/relative
    if relative.parts and relative.parts[0]=='models' and historical.exists() and (prefer_history or not active.exists()):
        return historical
    return active
