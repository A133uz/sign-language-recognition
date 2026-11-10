"""Copy into a new Colab cell AFTER colab_gpu_check_cell.py; experimental retry."""

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import urllib.request

if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('This cell targets the existing Linux x86-64 Colab runtime only.')
if not all(name in globals() for name in (
    'ENV_PYTHON', 'gpu_check_program', 'gpu_check_models', 'gpu_check_video',
)):
    raise RuntimeError('Run the GPU check cell first in this same notebook/runtime.')

# Exact official package and checksum verified on 2026-11-07. Never run its
# installer: --extract-only unpacks files without changing the system driver.
egl_version = '580.82.07'
egl_archive_name = f'NVIDIA-Linux-x86_64-{egl_version}-no-compat32.run'
egl_url = f'https://download.nvidia.com/XFree86/Linux-x86_64/{egl_version}/{egl_archive_name}'
egl_sha256 = 'c8a3c1306a6abb1d9856d966b42dc8ebd0d5f76275a34e59650d9c14e57cc0c4'
egl_drivers = set(subprocess.check_output(
    ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
    text=True, timeout=15,
).strip().splitlines())
if egl_drivers != {egl_version}:
    raise RuntimeError(f'Expected driver {egl_version}; found {egl_drivers}. Stop and report this; do not install another driver.')

egl_work = Path('/content') / f'asl-nvidia-egl-{egl_version}'
egl_work.mkdir(exist_ok=True)
egl_archive = egl_work / egl_archive_name
egl_libs = egl_work / 'lib'
egl_ready_marker = egl_work / 'graphics-libraries.json'


def egl_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def egl_libraries_ready():
    try:
        saved = json.loads(egl_ready_marker.read_text(encoding='utf-8'))
        hashes = saved['sha256']
        return (saved['driver'] == egl_version and bool(hashes)
                and all(Path(name).name == name and egl_hash(egl_libs / name) == digest
                        for name, digest in hashes.items()))
    except (OSError, ValueError, KeyError):
        return False


if not egl_archive.exists():
    if shutil.disk_usage(egl_work).free < 3 * 1024**3:
        raise RuntimeError('Need at least 3 GiB free on /content for the archive and temporary extraction.')
    print('Downloading the official NVIDIA archive (about 310 MB); no system installation.', flush=True)
    egl_partial = egl_archive.with_suffix('.run.part')
    with urllib.request.urlopen(egl_url, timeout=90) as response, egl_partial.open('wb') as target:
        downloaded, last_report = 0, 0
        while block := response.read(1024 * 1024):
            target.write(block)
            downloaded += len(block)
            if downloaded - last_report >= 32 * 1024**2:
                print(f'Downloaded {downloaded / 1024**2:.0f} MiB', flush=True)
                last_report = downloaded
    if egl_hash(egl_partial) != egl_sha256:
        raise RuntimeError('NVIDIA archive checksum mismatch; nothing was executed.')
    egl_partial.replace(egl_archive)
if egl_hash(egl_archive) != egl_sha256:
    raise RuntimeError('Cached archive checksum mismatch; nothing was executed.')

# Copy only graphics-side libraries. CUDA, NVML, kernel modules, the system
# linker cache and /usr driver files remain untouched.
if not egl_libraries_ready():
    if shutil.disk_usage(egl_work).free < 2 * 1024**3:
        raise RuntimeError('Need at least 2 GiB free for temporary archive extraction.')
    patterns = (
        'libEGL_nvidia.so.*', 'libGLESv2_nvidia.so.*', 'libGLESv1_CM_nvidia.so.*',
        'libnvidia-eglcore.so.*', 'libnvidia-glsi.so.*', 'libnvidia-glcore.so.*',
        'libnvidia-glvkspirv.so.*', 'libnvidia-allocator.so.*', 'libnvidia-tls.so.*',
    )
    with tempfile.TemporaryDirectory(prefix='unpack-', dir=egl_work) as temporary:
        unpacked = Path(temporary) / 'archive'
        print('Extracting files only; nvidia-installer will NOT run.', flush=True)
        subprocess.run(['sh', str(egl_archive), '--extract-only', '--target', str(unpacked)],
                       check=True, timeout=240)
        egl_libs.mkdir(exist_ok=True)
        for pattern in patterns:
            for candidate in unpacked.glob(pattern):
                if candidate.is_file():
                    shutil.copy2(candidate, egl_libs / candidate.name)
    for filename in (f'libEGL_nvidia.so.{egl_version}', f'libnvidia-eglcore.so.{egl_version}',
                     f'libnvidia-glsi.so.{egl_version}'):
        if not (egl_libs / filename).is_file():
            raise RuntimeError(f'Expected graphics library missing: {filename}')
    egl_ready_marker.write_text(json.dumps({
        'driver': egl_version,
        'sha256': {path.name: egl_hash(path) for path in egl_libs.glob(f'*.so.{egl_version}')},
    }), encoding='utf-8')

# Add the libraries' own SONAME aliases when readelf is present, otherwise the
# standard public NVIDIA EGL/GLES/allocator aliases. Everything stays private.
egl_aliases = {
    'libEGL_nvidia.so.0': f'libEGL_nvidia.so.{egl_version}',
    'libGLESv2_nvidia.so.2': f'libGLESv2_nvidia.so.{egl_version}',
    'libGLESv1_CM_nvidia.so.1': f'libGLESv1_CM_nvidia.so.{egl_version}',
    'libnvidia-allocator.so.1': f'libnvidia-allocator.so.{egl_version}',
}
if shutil.which('readelf'):
    for library in egl_libs.glob(f'*.so.{egl_version}'):
        info = subprocess.check_output(['readelf', '-d', str(library)], text=True, timeout=15)
        match = re.search(r'\(SONAME\).*?\[([^\]]+)\]', info)
        if match:
            egl_aliases[match.group(1)] = library.name
for alias, filename in egl_aliases.items():
    if (egl_libs / filename).exists() and not (egl_libs / alias).exists():
        (egl_libs / alias).symlink_to(filename)

egl_manifest = egl_work / '10_nvidia.json'
egl_manifest.write_text(json.dumps({
    'file_format_version': '1.0.0',
    'ICD': {'library_path': str(egl_libs / f'libEGL_nvidia.so.{egl_version}')},
}), encoding='utf-8')
egl_retry_env = {
    **os.environ, 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1',
    'TF_CPP_MIN_LOG_LEVEL': '0', 'GLOG_minloglevel': '0',
    '__EGL_VENDOR_LIBRARY_FILENAMES': str(egl_manifest),
    'LD_LIBRARY_PATH': str(egl_libs) + (':' + os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''),
}
egl_retry_env.pop('LIBGL_ALWAYS_SOFTWARE', None)
egl_retry_results = {}
for task, model in gpu_check_models.items():
    print(f'\n--- NVIDIA EGL retry: {task.upper()} ---', flush=True)
    try:
        probe = subprocess.run(
            [str(ENV_PYTHON), '-u', '-c', gpu_check_program, task, str(model), str(gpu_check_video)],
            env=egl_retry_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', timeout=120,
        )
        log = probe.stdout or '(No diagnostic output.)'
        passed = probe.returncode == 0 and any(line.startswith('PROBE_RESULT=') for line in log.splitlines())
        egl_retry_results[task] = {'status': 'GPU_REQUEST_PASS' if passed else 'FAIL',
                                   'exit_code': probe.returncode, 'log': log}
        print(log, flush=True)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b''
        log = output.decode(errors='replace') if isinstance(output, bytes) else output
        egl_retry_results[task] = {'status': 'TIMEOUT', 'log': log}
        print(log, flush=True)
    print(task.upper() + ': ' + egl_retry_results[task]['status'], flush=True)

print('\nEGL RETRY SUMMARY:', {name: item['status'] for name, item in egl_retry_results.items()})
print('Check for an NVIDIA/Tesla renderer plus successful inference. Paste the complete output here.')
print('Only retry children received the private library settings. Full extraction still uses CPU; no training ran.')
