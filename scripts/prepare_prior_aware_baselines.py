"""Pin official baseline sources and download Pow3R's published checkpoint locally."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen
from evaluation_stability import dump
from prepare_dense_focal_pilot import ROOT

URL='https://download.europe.naverlabs.com/ComputerVision/Pow3R/Pow3R_ViTLarge_BaseDecoder_512_linear.pth'
SIZE=2223286338
SOURCES={'third_party/pow3r':'25d7bb68e01336585b5277d5d6aafadddf9e1b8d',
         'third_party/pow3r/dust3r':'ad9d1a0421b19d7be72801f6e2e8bfcf79dc3c68',
         'third_party/pow3r/dust3r/croco':'d7de0705845239092414480bd829228723bf20de',
         'third_party/Test3R':'a2eb94bc716df27521f053d417fcf6afa9870ff5'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def prepare(destination):
    for name,expected in SOURCES.items():
        actual=subprocess.check_output(['git','-C',str(ROOT/name),'rev-parse','HEAD'],text=True).strip()
        assert actual==expected,(name,actual)
    destination.mkdir(parents=True,exist_ok=True)
    target=destination/'Pow3R_ViTLarge_BaseDecoder_512_linear.pth'
    record_path=ROOT/'results/prior_aware_baselines/preparation.json'
    start=time.perf_counter()
    if target.exists():
        from score_saved_geometry import read
        old=read(record_path)
        assert target.stat().st_size==SIZE and digest(target)==old['pow3r_checkpoint']['sha256']
        print('Existing baseline checkpoint verified.');return
    temporary=target.with_suffix('.pth.partial')
    if temporary.exists():raise ValueError('An incomplete download is preserved; review it before retrying')
    with urlopen(URL,timeout=30) as response,temporary.open('wb') as f:
        assert int(response.headers['Content-Length'])==SIZE
        received=0;mark=0
        for block in iter(lambda:response.read(8*1024**2),b''):
            f.write(block);received+=len(block)
            if received-mark>=256*1024**2:
                print('Pow3R download:',round(received/1024**2),'MiB',flush=True);mark=received
        assert received==SIZE
    checksum=digest(temporary);temporary.replace(target)
    dump(record_path,dict(source_commits=SOURCES,pow3r_checkpoint=dict(url=URL,bytes=SIZE,sha256=checksum,
        hash_provenance='Locally computed after official-vendor HTTPS download; no upstream digest supplied'),
        download_seconds=time.perf_counter()-start,additional_runpod_compute_usd=0,
        status='source_and_checkpoint_preparation_only_no_gpu_inference_or_quality_evaluation'))
    print('Pow3R checkpoint downloaded and pinned:',checksum,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True)
    prepare(p.parse_args().destination)
