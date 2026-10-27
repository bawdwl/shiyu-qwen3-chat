"""Download the exact official wheel in resumable ranges for local reproduction."""
import hashlib
import sys
import html
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlsplit, unquote
import httpx

INDEX='https://download.pytorch.org/whl/cu126/torch/'
class Links(HTMLParser):
    def __init__(self): super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a': self.links.extend(value for key,value in attrs if key=='href')

def main():
    if sys.platform!='win32' or sys.version_info[:2]!=(3,11):
        raise RuntimeError('This downloader is for Windows x64 / Python 3.11 only')
    response=httpx.get(INDEX,timeout=30);response.raise_for_status()
    parser=Links();parser.feed(response.text)
    matches=[urljoin(INDEX,html.unescape(u)) for u in parser.links
             if 'torch-2.13.0+cu126-cp311-cp311-win_amd64.whl' in unquote(u)]
    if len(matches)!=1: raise RuntimeError('Cannot identify exact official wheel: '+str(len(matches)))
    link=matches[0];url,fragment=link.split('#',1)
    expected=fragment.removeprefix('sha256=')
    if len(expected)!=64 or urlsplit(url).hostname not in ('download.pytorch.org','download-r2.pytorch.org'):
        raise RuntimeError('Unexpected official wheel metadata')
    base=Path(__file__).resolve().parent/'outputs/repro-wheels';base.mkdir(parents=True,exist_ok=True)
    filename=unquote(urlsplit(url).path.rsplit('/',1)[1]);target=base/filename
    if target.exists():
        digest=hashlib.sha256()
        with target.open('rb') as source:
            for chunk in iter(lambda:source.read(8*1024*1024),b''): digest.update(chunk)
        if digest.hexdigest()==expected:
            print('VERIFIED '+str(target),flush=True);return
    with httpx.stream('GET',url,headers={'Range':'bytes=0-0'},follow_redirects=True,timeout=30) as r:
        r.raise_for_status();size=int(r.headers['Content-Range'].split('/')[-1])
    block=1024*1024;count=(size+block-1)//block
    directory=base/'torch-parts-1m';directory.mkdir(exist_ok=True)
    print(f'Official wheel {size} bytes, {count} ranges, sha256={expected}',flush=True)
    def fetch(index):
        start=index*block;end=min(size,start+block)-1
        path=directory/f'{index:05d}'
        if path.exists() and path.stat().st_size==end-start+1: return path
        for attempt in range(3):
            try:
                with client.stream('GET',url,headers={'Range':f'bytes={start}-{end}'}) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                        raise ValueError('Range response mismatch')
                    content=r.read()
                partial=path.with_suffix('.part')
                partial.write_bytes(content)
                if partial.stat().st_size!=end-start+1: raise ValueError('Truncated range')
                partial.replace(path);return path
            except (httpx.HTTPError,ValueError):
                if attempt==2: raise
    with httpx.Client(follow_redirects=True,timeout=30) as client, ThreadPoolExecutor(max_workers=8) as pool:
        done=0
        for batch in range(0,count,8):
            jobs=[pool.submit(fetch,index) for index in range(batch,min(batch+8,count))]
            for job in as_completed(jobs):
                job.result();done+=1
            if done%32==0 or done==count: print(f'Ranges {done}/{count}',flush=True)
    digest=hashlib.sha256();partial=target.with_suffix('.whl.part')
    with partial.open('wb') as out:
        for index in range(count):
            with (directory/f'{index:05d}').open('rb') as source:
                for chunk in iter(lambda:source.read(1024*1024),b''):
                    digest.update(chunk);out.write(chunk)
    if digest.hexdigest()!=expected: raise ValueError('Official wheel SHA-256 mismatch')
    partial.replace(target)
    print('VERIFIED '+str(target),flush=True)

if __name__=='__main__': main()
