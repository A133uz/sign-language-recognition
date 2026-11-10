"""Generate a single copy/paste setup cell and notebook with an embedded source snapshot."""

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    files = [Path('requirements-colab.txt'), Path('scripts/project.py')]
    files += [Path('src/sign_language') / name for name in (
        '__init__.py', 'paths.py', 'cli.py', 'dataset.py', 'assets.py', 'hands.py',
        'webcam.py', 'landmark_models.py', 'raw_landmarks.py', 'extraction.py',
    )]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative in sorted(files):
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, (ROOT / relative).read_text(encoding='utf-8').encode())
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    source = (ROOT/'notebooks/colab_setup_template.py').read_text(encoding='utf-8')
    source = source.replace('__SOURCE_BUNDLE_BASE64__', base64.b64encode(payload).decode())
    source = source.replace('__SOURCE_BUNDLE_SHA256__', digest)
    cache = ROOT / '.cache/colab-delivery'
    cache.mkdir(parents=True, exist_ok=True)
    (cache/'asl_citizen_extraction_cell.py').write_text(source, encoding='utf-8', newline='\n')
    (cache/'asl_citizen_extractor_source.zip').write_bytes(payload)
    gpu_check = (ROOT/'notebooks/colab_gpu_check_cell.py').read_text(encoding='utf-8')
    (cache/'asl_citizen_gpu_check_cell.py').write_text(gpu_check, encoding='utf-8', newline='\n')
    egl_retry = (ROOT/'notebooks/colab_egl_retry_cell.py').read_text(encoding='utf-8')
    (cache/'asl_citizen_egl_retry_cell.py').write_text(egl_retry, encoding='utf-8', newline='\n')
    comparison = (ROOT/'notebooks/colab_compare_delegates_cell.py').read_text(encoding='utf-8')
    (cache/'asl_citizen_compare_delegates_cell.py').write_text(comparison, encoding='utf-8', newline='\n')
    audit = (ROOT/'notebooks/colab_audit_comparison_cell.py').read_text(encoding='utf-8')
    (cache/'asl_citizen_audit_comparison_cell.py').write_text(audit, encoding='utf-8', newline='\n')
    markdown = (
        '# ASL Citizen landmark extraction only\n\n'
        'If you already downloaded/unzipped the dataset, keep that existing notebook/runtime. '
        'Copy the setup cell below into it; opening a separate runtime may lose access to your current `/content` data.\n\n'
        'The setup cell installs an isolated environment, mounts your private Google Drive, and runs '
        'a ten-clip hand-and-body pilot. Inspect its results before the separate full-run cell. '
        'The same source snapshot, settings and output folder resume completed checksummed batches. '
        'This is CPU landmark extraction, not T4 training. No classifier training occurs.\n'
    )
    def cell(kind, text):
        data = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True),
                'id': hashlib.sha256((kind + text).encode()).hexdigest()[:12]}
        if kind == 'code':
            data.update(execution_count=None, outputs=[])
        return data
    notebook = {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [cell('markdown', markdown), cell('code', source),
                  cell('markdown', '## Inspect the pilot\n\nRead `summary.json` and the tracking review instructions in `docs/landmark-extraction.md`. '
                       'Zero failures alone does not prove correct hand tracking. Keep outputs private.\n'),
                  cell('code', 'inspect_pilot()\n'),
                  cell('markdown', '## Optional GPU compatibility check\n\n'
                       'Copy `notebooks/colab_gpu_check_cell.py` into a new cell after the setup/pilot '
                       'to test hands and pose with explicit GPU requests and native diagnostics. '
                       'It does not change the CPU extractor or run training.\n'),
                  cell('markdown', '## Full extraction (run explicitly after the pilot)\n\n'
                       'Run once per output folder. It processes the official train/val/test splits and resumes valid shards. '
                       'A stopped, uncommitted batch is recomputed; completed batches remain on Drive.\n'),
                  cell('code', '# Uncomment only after reviewing the pilot:\n# result = run_full_extraction()\n# print(json.dumps(result, indent=2))\n'),
                  cell('markdown', '## Resume or investigate failures\n\n'
                       'After VM loss, restore the raw dataset, then run the SAME setup cell and full-run call. '
                       'Changing schema/models/library versions requires a separate cache. '
                       'Inspect failures before explicitly retrying their shards with `retry_failed=True`. '
                       'The model trainer still needs implementation and separate authorization.\n')],
    }
    path = ROOT/'notebooks/asl_citizen_landmark_extraction.ipynb'
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({'source_sha256': digest, 'source_zip_bytes': len(payload),
                      'setup_cell': str(cache/'asl_citizen_extraction_cell.py'),
                      'gpu_check_cell': str(cache/'asl_citizen_gpu_check_cell.py'),
                      'egl_retry_cell': str(cache/'asl_citizen_egl_retry_cell.py'),
                      'comparison_cell': str(cache/'asl_citizen_compare_delegates_cell.py'),
                      'audit_cell': str(cache/'asl_citizen_audit_comparison_cell.py'),
                      'notebook': str(path)}, indent=2))


if __name__ == '__main__':
    build()
