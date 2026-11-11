"""Create/verify a complete, immutable local model archive with per-file SHA256.

Includes selected, last and per-epoch weights, attempted/smoke runs, configs,
labels, reports, model source snapshots and MediaPipe tasks. Original files stay
in place. Data, webcam media, old archives and locks are excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import stat
import subprocess
from zipfile import ZipFile,ZipInfo,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
CHUNK=1024*1024


def digest_file(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def safe_member(name):
    path=PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or '..' in path.parts or any(':' in x for x in path.parts):
        raise ValueError('Unsafe archive member name')
    return path


def inventory(root):
    root=Path(root).resolve();models=root/'models';files={}
    for current,dirs,names in os.walk(models):
        dirs[:]=sorted(d for d in dirs if d not in ('archives','locks','__pycache__','.git'))
        for name in sorted(names):
            path=Path(current)/name
            if path.suffix.lower() in ('.pyc','.pyo','.lock','.tmp','.part'):continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Model path leaves workspace')
            files[path.relative_to(root).as_posix()]=path
    for folder in ('src','scripts','tests'):
        for path in sorted((root/folder).rglob('*.py')):
            if '__pycache__' in path.parts:continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):raise ValueError('Source path leaves workspace')
            files['project_source/'+path.relative_to(root).as_posix()]=path
    for name in ('pyproject.toml','uv.lock','.gitattributes','.gitignore'):
        path=root/name
        if path.is_file():files['project_source/'+name]=path
    if not any(name.endswith(('.pt','.pth')) for name in files):raise ValueError('No checkpoints found')
    return dict(sorted(files.items()))


def verify_archive(path,expected_sha256=None):
    path=Path(path)
    if expected_sha256 and digest_file(path)!=expected_sha256:raise ValueError('Archive SHA256 differs')
    with ZipFile(path) as archive:
        infos=archive.infolist();names=[i.filename for i in infos]
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive members')
        for info in infos:
            safe_member(info.filename)
            if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Archive contains a symlink')
        manifest=json.loads(archive.read('MANIFEST.json'))
        expected={r['member']:r for r in manifest['files']}
        if len(expected)!=len(manifest['files']) or set(names)!=set(expected)|{'MANIFEST.json'}:
            raise ValueError('Manifest does not cover archive members exactly')
        for name,record in expected.items():
            info=archive.getinfo(name)
            if info.file_size!=record['bytes']:raise ValueError('Archive member size differs: '+name)
            h=hashlib.sha256()
            with archive.open(name) as stream:
                while data:=stream.read(CHUNK):h.update(data)
            # Reading each complete member also verifies its ZIP CRC.
            if h.hexdigest()!=record['sha256']:raise ValueError('Archive member SHA256 differs: '+name)
    return manifest


def checkpoint_catalog(files,records):
    by_name={r['member']:r for r in records};result=[]
    for member,path in files.items():
        if not member.startswith('models/') or path.name!='best.pt':continue
        entry={'path':member,'sha256':by_name[member]['sha256']}
        for name in ('config.json','status.json','selection.json'):
            candidate=path.with_name(name)
            if not candidate.exists():continue
            try:value=json.loads(candidate.read_text(encoding='utf-8'))
            except (ValueError,OSError):entry[name]='unreadable metadata preserved in archive';continue
            if name=='config.json':entry.update(architecture=value.get('architecture','gru'),model=value.get('model'),cache_id=value.get('cache_id'))
            elif name=='status.json':entry.update(status=value.get('status'),epoch=value.get('epoch'),epochs_completed=value.get('epochs_completed'))
            else:entry['selected_epoch']=value.get('selection',{}).get('selected_epoch')
        result.append(entry)
    return result


RESTORE='''# Model archive restoration

This archive preserves all retained model runs, including diagnostic, smoke,
failed/incomplete attempts and per-epoch files. Inclusion is not endorsement.
models/ paths restore original configs/labels/source snapshots alongside weights.
project_source/ is a separate code/dependency snapshot; it is not an instruction
to overwrite current code. Dataset videos and landmark caches are not included.

1. Verify the ZIP SHA256 against its .sha256 file / published catalog.
2. Run: python scripts/archive_model_weights.py --verify PATH_TO_ARCHIVE
3. Expand into a NEW empty staging directory, not over existing model files.
4. Copy the chosen models/<run> directory into a new local model directory.
   Keep best.pt, config.json, labels.json and source snapshots together.
5. Copy the MediaPipe .task files from models/ if they are not already installed.
6. Use scripts/recognize_webcam.py --checkpoint models/<chosen-run>/best.pt.

Runtime source-hash guards must match the archived configuration. Use the saved
source snapshot for historical runs if required. Training resumption additionally
requires the matching dataset/feature cache, which remains outside this archive.

No training data or webcam imagery was added to this archive. Original model
files were preserved. Verify MANIFEST.json for exact members and SHA256 values.
'''


def create_archive(root,output,catalog_path):
    root=Path(root).resolve();output=Path(output).resolve();catalog_path=Path(catalog_path)
    archive_root=(root/'models/archives').resolve()
    if not output.is_relative_to(archive_root):raise ValueError('Archive output must remain inside models/archives')
    partial=output.with_suffix(output.suffix+'.partial')
    if output.exists() or partial.exists():raise FileExistsError('Archive/partial exists; use a new version')
    files=inventory(root);total=sum(p.stat().st_size for p in files.values())
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free<total+128*1024**2:raise OSError('Insufficient free space for conservative archive budget')
    base_commit=subprocess.check_output(['git','-c',f'safe.directory={root.as_posix()}','rev-parse','HEAD'],cwd=root,text=True).strip()
    records=[]
    with partial.open('xb') as stream,ZipFile(stream,'w',compression=ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for number,(member,path) in enumerate(files.items(),1):
            safe_member(member);before=path.stat();h=hashlib.sha256();info=ZipInfo.from_file(path,arcname=member)
            info.compress_type=ZIP_DEFLATED
            with path.open('rb') as source,archive.open(info,'w',force_zip64=True) as destination:
                while data:=source.read(CHUNK):h.update(data);destination.write(data)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source changed while archiving: '+member)
            records.append({'member':member,'bytes':after.st_size,'sha256':h.hexdigest()})
            if number%100==0:print(f'ARCHIVE_WRITE {number}/{len(files)} files',flush=True)
        restore=RESTORE.encode('utf-8');archive.writestr('README_RESTORE.md',restore)
        records.append({'member':'README_RESTORE.md','bytes':len(restore),'sha256':hashlib.sha256(restore).hexdigest()})
        manifest={'schema':'sign-language-model-archive-v1','archive_date_local':'2026-11-12','timezone':'Asia/Tashkent',
            'base_git_commit':base_commit,'git_worktree_snapshot':True,'scope':'Complete retained models tree plus project Python/dependency sources',
            'exclusions':['models/archives','models/locks','__pycache__','compiled/temporary files','datasets/features','webcam media'],
            'checkpoint_files':sum(r['member'].startswith('models/') and r['member'].endswith(('.pt','.pth')) for r in records),
            'mediapipe_task_files':sum(r['member'].endswith('.task') for r in records),
            'selected_checkpoint_catalog':checkpoint_catalog(files,records),'files':records,
            'limits':'Includes intermediate and failed/diagnostic runs for preservation; no model quality or deployment-readiness claim. MANIFEST.json is the only self-excluded member.'}
        manifest_bytes=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode('utf-8');archive.writestr('MANIFEST.json',manifest_bytes)
    print('ARCHIVE_VERIFY reading every member for CRC and SHA256',flush=True)
    verify_archive(partial)
    if any(digest_file(files[r['member']])!=r['sha256'] for r in records if r['member'] in files):raise AssertionError('Original source bytes changed')
    if output.exists():raise FileExistsError(output)
    partial.rename(output)
    archive_sha=digest_file(output);output.with_suffix(output.suffix+'.sha256').write_text(archive_sha+'  '+output.name+'\n',encoding='ascii')
    catalog={'archive_path':output.relative_to(root).as_posix(),'archive_sha256':archive_sha,
        'archive_bytes':output.stat().st_size,'uncompressed_payload_bytes':sum(r['bytes'] for r in records),
        'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'verified_crc_and_member_sha256':True,
        'original_sources_unchanged':True,'archive_uploaded_to_git':False,'manifest':manifest}
    catalog_path.parent.mkdir(parents=True,exist_ok=True);catalog_path.write_text(json.dumps(catalog,indent=2)+'\n',encoding='utf-8')
    print('ARCHIVE_COMPLETE',json.dumps({k:v for k,v in catalog.items() if k!='manifest'}),flush=True)
    return catalog


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'models/archives/all-model-checkpoints-2026-11-12.zip')
    p.add_argument('--catalog',type=Path,default=ROOT/'docs/assets/model-archive-2026-11-12-manifest.json')
    a=p.parse_args()
    if a.verify:
        m=verify_archive(a.verify);print('ARCHIVE_VERIFIED',len(m['files']),'payload members',m['checkpoint_files'],'checkpoint files')
    else:create_archive(ROOT,a.output,a.catalog)


if __name__=='__main__':main()
