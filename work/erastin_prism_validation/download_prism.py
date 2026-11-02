import concurrent.futures,json,hashlib,time,urllib.request
from pathlib import Path
out=Path(__file__).resolve().parent/'raw';a=json.load(open(out/'figshare_20564034.json'))
selected=[f for f in a['files'] if any(s in f['name'] for s in ['secondary-screen-dose-response-curve-parameters','secondary-screen-cell-line-info','secondary-screen-readme','secondary-screen-replicate-collapsed-treatment-info','primary-screen-replicate-collapsed-treatment-info','primary-screen-readme'])]
def one(f):
    p=out/f['name'];start=time.monotonic()
    if not p.exists():
        with urllib.request.urlopen(f['download_url'],timeout=60) as r,open(p.with_suffix(p.suffix+'.partial'),'wb') as w:
            while True:
                chunk=r.read(1<<20)
                if not chunk:break
                w.write(chunk)
        p.with_suffix(p.suffix+'.partial').replace(p)
    data=p.read_bytes();assert len(data)==f['size'] and hashlib.md5(data).hexdigest()==f['computed_md5']
    rec=dict(name=f['name'],url=f['download_url'],bytes=len(data),md5=hashlib.md5(data).hexdigest(),sha256=hashlib.sha256(data).hexdigest(),download_seconds=time.monotonic()-start,retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),article_doi=a['doi'])
    (out/(p.name+'.download.json')).write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True);return rec
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,selected))
(out/'download_manifest.json').write_text(json.dumps(dict(files=results),indent=2))
