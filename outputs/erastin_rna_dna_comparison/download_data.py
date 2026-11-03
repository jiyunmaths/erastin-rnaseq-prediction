#!/usr/bin/env python3
"""Download only the pinned official DNA matrices, with size/MD5/SHA256 checks."""
import concurrent.futures as cf
import hashlib
import json
import threading
import time
import urllib.request
from pathlib import Path

WS=Path(__file__).resolve().parents[2]
RAW=WS/'work/erastin_rna_dna/raw'
NAMES=['OmicsSomaticMutationsMatrixHotspot.csv','OmicsSomaticMutationsMatrixDamaging.csv','OmicsCNGene.csv']
LOCK=threading.Lock();STATUS={}


def digest(path,algorithm):
    with path.open('rb') as f:return hashlib.file_digest(f,algorithm).hexdigest()


def fetch(record):
    path=RAW/record['name'];partial=path.with_suffix(path.suffix+'.part');start=time.monotonic()
    if path.exists() and path.stat().st_size==record['size'] and digest(path,'md5')==record['computed_md5']:
        result=dict(name=record['name'],url=record['download_url'],bytes=record['size'],md5=record['computed_md5'],sha256=digest(path,'sha256'),reused=True)
    else:
        for attempt in range(4):
            try:
                have=partial.stat().st_size if partial.exists() else 0
                headers={'User-Agent':'erastin-rnaseq-research/1.0'}
                if have:headers['Range']=f'bytes={have}-'
                req=urllib.request.Request(record['download_url'],headers=headers)
                with urllib.request.urlopen(req,timeout=90) as response:
                    append=have>0 and response.status==206
                    if not append:have=0
                    with partial.open('ab' if append else 'wb') as out:
                        mark=time.monotonic()
                        while True:
                            chunk=response.read(4*1024*1024)
                            if not chunk:break
                            out.write(chunk);have+=len(chunk)
                            assert have<=record['size']
                            if time.monotonic()-mark>15:
                                print(json.dumps(dict(file=record['name'],bytes=have,total=record['size'],seconds=time.monotonic()-start)),flush=True)
                                mark=time.monotonic()
                assert partial.stat().st_size==record['size'],(record['name'],partial.stat().st_size)
                assert digest(partial,'md5')==record['computed_md5'],record['name']
                partial.replace(path);break
            except Exception:
                if attempt==3:raise
                # A completed but invalid file must never be appended to on retry.
                if partial.exists() and partial.stat().st_size>=record['size']:partial.unlink()
                time.sleep(2*(attempt+1))
        result=dict(name=record['name'],url=record['download_url'],bytes=record['size'],md5=record['computed_md5'],sha256=digest(path,'sha256'),reused=False)
    result['elapsed_seconds']=time.monotonic()-start
    with LOCK:
        STATUS[record['name']]=result
        (RAW/'download_progress.json').write_text(json.dumps(dict(completed=STATUS,total=3),indent=2)+'\n')
    print(json.dumps(result),flush=True);return result


def main():
    RAW.mkdir(parents=True,exist_ok=True)
    cached=json.load(open(WS/'work/erastin/raw/depmap_24q4_article.json'))
    assert cached['version']==1
    lookup={r['name']:r for r in cached['files']}
    # Confirm selected cached file identifiers, checksums and sizes against live metadata.
    with urllib.request.urlopen('https://api.figshare.com/v2/articles/27993248/versions/1',timeout=40) as r:live=json.load(r)
    current={r['name']:r for r in live['files']}
    for name in NAMES:
        for key in ['id','size','computed_md5']:assert lookup[name][key]==current[name][key],(name,key)
    (RAW/'release_selected_files.json').write_text(json.dumps(dict(article_id=27993248,version=1,doi=cached.get('doi'),files=[lookup[n] for n in NAMES]),indent=2)+'\n')
    with cf.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,[lookup[n] for n in NAMES]))
    (RAW/'download_manifest.json').write_text(json.dumps(dict(release='DepMap 24Q4 Public v1',official_metadata_verified_live=True,files=rows),indent=2)+'\n')
    print('DNA_DOWNLOADS_COMPLETE',flush=True)


if __name__=='__main__':main()
