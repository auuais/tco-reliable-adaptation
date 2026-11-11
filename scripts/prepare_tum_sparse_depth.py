"""Download the fixed public TUM archives and validate gzip integrity without extracting files."""
import argparse
import hashlib
from pathlib import Path
import tarfile

from evaluation_stability import dump,sha
from prepare_dense_focal_pilot import ROOT

SEQUENCES=('long_office_household','structure_texture_far','nostructure_texture_near_withloop')
PUBLIC=ROOT/'results/tum_sparse_depth'


def download(destination):
    import requests
    destination.mkdir(parents=True,exist_ok=True)
    for sequence in SEQUENCES:
        name='rgbd_dataset_freiburg3_'+sequence+'.tgz'
        url='https://cvg.cit.tum.de/rgbd/dataset/freiburg3/'+name
        final=destination/name
        partial=final.with_suffix('.partial')
        if not final.exists():
            offset=partial.stat().st_size if partial.exists() else 0
            headers={'Range':f'bytes={offset}-'} if offset else {}
            with requests.get(url,headers=headers,stream=True,timeout=(20,90)) as response:
                response.raise_for_status()
                if offset and response.status_code==206:
                    if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise ValueError('Server resumed from a different offset')
                    mode='ab'
                else:
                    mode='wb'
                with partial.open(mode) as stream:
                    for block in response.iter_content(4*1024*1024):
                        stream.write(block)
            # Consume the complete compressed stream, including the CRC-protected trailer.
            import gzip
            with gzip.open(partial,'rb') as stream:
                while stream.read(4*1024*1024):
                    pass
            partial.replace(final)
        members=[]
        metadata={}
        prefix=name[:-4]+'/'
        with tarfile.open(final,'r:gz') as archive:
            for item in archive:
                path=Path(item.name)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe archive member')
                if item.isfile():
                    members.append(item.name)
                    if item.name in {prefix+s for s in ('rgb.txt','depth.txt','groundtruth.txt') }:
                        with archive.extractfile(item) as stream:
                            metadata[path.name]=stream.read()
        if set(metadata)!={'rgb.txt','depth.txt','groundtruth.txt'}:
            raise ValueError('Missing timestamp or trajectory index')
        folder=destination/name[:-4]
        folder.mkdir(exist_ok=True)
        for filename,value in metadata.items():
            path=folder/filename
            if path.exists() and path.read_bytes()!=value:
                raise ValueError('Existing timestamp metadata changed')
            path.write_bytes(value)
        record=dict(sequence=sequence,url=url,archive_bytes=final.stat().st_size,archive_sha256=sha(final),
                    files=len(members),metadata_sha256={k:hashlib.sha256(v).hexdigest() for k,v in metadata.items()})
        path=PUBLIC/(sequence+'_download.json')
        if path.exists():
            import json
            if json.loads(path.read_text())!=record:
                raise ValueError('Archive identity changed')
        dump(path,record)
        print(sequence,'gzip validated and timestamp metadata staged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    download(parser.parse_args().destination)
