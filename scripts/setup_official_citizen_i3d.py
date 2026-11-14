"""Install the pinned official baseline sources/weights; preserve existing runs."""
import csv
import json
from pathlib import Path
import sys
from urllib.request import urlretrieve
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sign_language.five_sign_pilot import sha256,write_json


def fetch(url,path,expected):
    if path.exists():
        if sha256(path)!=expected:raise ValueError('Existing artifact hash differs: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.download');urlretrieve(url,temporary)
    if sha256(temporary)!=expected:raise ValueError('Downloaded artifact checksum differs: '+str(path))
    temporary.replace(path)


def main():
    spec=json.loads((ROOT/'docs/assets/official-citizen-i3d-install-spec-2026-11-14.json').read_text())
    code=(ROOT/spec['upstream_directory']).resolve();folder=ROOT/'models/asl-citizen-official-i3d-v1'
    if not code.is_relative_to(ROOT):raise ValueError('Source destination outside workspace')
    for name,h in spec['source_sha256'].items():
        path=(code/name).resolve()
        if not path.is_relative_to(code):raise ValueError('Unsafe upstream file path')
        fetch(f'https://raw.githubusercontent.com/microsoft/ASL-citizen-code/{spec["commit"]}/{name}',path,h)
    folder.mkdir(exist_ok=True)
    archive=folder/'ASL_citizen_I3D_weights.zip'
    weights=folder/'ASL_citizen_I3D_weights.pt'
    if not weights.exists():
        fetch(spec['release_url'],archive,spec['zip_sha256'])
        with zipfile.ZipFile(archive) as z:payload=z.read(weights.name)
        import hashlib
        if hashlib.sha256(payload).hexdigest()!=spec['checkpoint_sha256']:raise ValueError('Checkpoint checksum differs')
        weights.write_bytes(payload)
    if sha256(weights)!=spec['checkpoint_sha256']:raise ValueError('Existing official checkpoint differs')
    train=ROOT/'data/asl_citizen/ASL_Citizen/splits/train.csv'
    if sha256(train)!=spec['train_csv_sha256']:raise ValueError('Requires matching official training CSV')
    with train.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.reader(stream);next(reader);names=sorted({r[2].strip() for r in reader})
    if len(names)!=2731:raise ValueError('Unexpected official vocabulary')
    for path,value in [(folder/'config.json',spec),(folder/'labels.json',{g:i for i,g in enumerate(names)})]:
        if path.exists():
            if json.loads(path.read_text())!=value:raise ValueError('Preserve differing existing metadata: '+str(path))
        else:write_json(path,value)
    print('OFFICIAL_I3D_READY',spec['checkpoint_sha256'],'| no training or existing model changes')


if __name__=='__main__':main()
