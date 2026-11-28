"""Read individual members of a publisher ZIP64 scene without whole-archive fallback."""
import hashlib
import io
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile
import zlib


class RangeFile(io.RawIOBase):
    def __init__(self,url):
        super().__init__()
        if not re.fullmatch(r"https://docs-assets\.developer\.apple\.com/ml-research/datasets/hypersim/v1/scenes/ai_\d{3}_\d{3}\.zip",url):
            raise ValueError("Unexpected scene archive host/path")
        self.url=url
        with urlopen(Request(url,method="HEAD",headers={"Accept-Encoding":"identity"}),timeout=30) as response:
            self.length=int(response.headers["Content-Length"])
            self.etag=response.headers.get("ETag")
        if not self.etag or self.length<=0:
            raise ValueError("Publisher must identify the immutable byte range source")
        self.pos=0
        self.received=0
        self.requests=0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self,offset,whence=0):
        new={0:0,1:self.pos,2:self.length}[whence]+offset
        if new<0:
            raise OSError("Range before archive")
        self.pos=new
        return new

    def read(self,n=-1):
        size=max(0,self.length-self.pos) if n<0 else max(0,min(n,self.length-self.pos))
        if not size:
            return b""
        if size>32*1024**2 or self.received+size>350*1024**2:
            raise ValueError("Selective archive request exceeded registered resource bound")
        end=self.pos+size-1
        headers={"Range":f"bytes={self.pos}-{end}","If-Match":self.etag,"Accept-Encoding":"identity"}
        for trial in range(3):
            try:
                with urlopen(Request(self.url,headers=headers),timeout=45) as response:
                    if (response.status!=206 or response.headers.get("Content-Range")!=f"bytes {self.pos}-{end}/{self.length}"
                            or response.headers.get("ETag")!=self.etag):
                        raise ValueError("Server did not preserve the requested archive range and ETag")
                    payload=response.read(size+1)
                break
            except (URLError,TimeoutError) as exc:
                if isinstance(exc,HTTPError) and exc.code not in (408,429,500,502,503,504):
                    raise
                if trial==2:
                    raise
                time.sleep(1+trial)
        if len(payload)!=size:
            raise ValueError("Truncated archive range")
        self.requests+=1
        self.received+=size
        self.pos=end+1
        return payload


class SceneArchive:
    def __init__(self,url,folder):
        self.remote=RangeFile(url)
        self.folder=folder
        self.zip=zipfile.ZipFile(self.remote)
        self.names=set(self.zip.namelist())
        if len(self.names)!=len(self.zip.infolist()):
            raise ValueError("Ambiguous duplicated scene archive member")

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.zip.close()
        self.remote.close()

    def member(self,key):
        if key.startswith("/") or any(k in ("", ".", "..") for k in key.split("/")) or "\\" in key:
            raise ValueError("Invalid scene archive member path")
        info=self.zip.getinfo(key)
        if info.file_size>64*1024**2 or info.compress_size>32*1024**2:
            raise ValueError("Unexpected scene member size")
        p=self.folder.joinpath(*key.split("/"))
        if p.is_file():
            raw=p.read_bytes()
        else:
            raw=self.zip.read(info)
            p.parent.mkdir(exist_ok=True,parents=True)
            partial=p.with_suffix(p.suffix+".partial")
            partial.write_bytes(raw)
            partial.replace(p)
        if len(raw)!=info.file_size or zlib.crc32(raw)!=info.CRC:
            raise ValueError("Archive member CRC differs from publisher ZIP")
        return raw,hashlib.sha256(raw).hexdigest()
