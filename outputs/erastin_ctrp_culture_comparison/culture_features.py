"""Culture covariates with training-only category grouping and continuous scaling."""
import re
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
CAT_FIELDS=['growth_pattern','medium_code','basal_medium']
MIN_COUNTS={'growth_pattern':1,'medium_code':5,'basal_medium':5}
MISSING='__MISSING__';OTHER='__OTHER__'

def extract_metadata(models,ids):
    rows=[]
    for mid in ids:
        m=models.loc[mid];growth=m.GrowthPattern.strip();code=m.OnboardedMedia.strip();recipe=m.FormulationID.strip()
        # Preserve this release's actual column contents: MF code under OnboardedMedia,
        # recipe text under FormulationID. The accompanying README labels these oppositely.
        assert (not code and not recipe) or (re.fullmatch(r'MF-\d{3}-\d{3}',code) and recipe and not recipe.startswith('MF-'))
        match=re.search(r'(\d+(?:\.\d+)?)\s*%\s*FBS\b',recipe,flags=re.I)
        assert not recipe or match is not None or m.SerumFreeMedia=='True'
        # An explicit serum-free flag does not supply a numeric FBS percentage.
        # Preserve that unreported value as missing rather than inferring zero.
        serum=float(match.group(1)) if match else np.nan
        rows.append(dict(ModelID=str(mid),growth_pattern=MISSING if growth.lower() in ['','unknown'] else growth,medium_code=code or MISSING,basal_medium=recipe.split('+',1)[0].strip() if recipe else MISSING,serum_percent=serum,serum_missing=not np.isfinite(serum),raw_GrowthPattern=m.GrowthPattern,raw_OnboardedMedia=m.OnboardedMedia,raw_FormulationID=m.FormulationID,raw_SerumFreeMedia=m.SerumFreeMedia,raw_PlateCoating=m.PlateCoating))
    d=pd.DataFrame(rows);assert d.ModelID.is_unique
    known=d[d.medium_code!=MISSING];assert known.groupby('medium_code').raw_FormulationID.nunique().max()==1 and known.groupby('raw_FormulationID').medium_code.nunique().max()==1
    return d

class CulturePreprocessor:
    """Standardize genes/serum; keep centered categorical indicators unscaled."""
    def fit(self,x,meta):
        x=np.asarray(x,float);assert len(x)==len(meta);self.n_genes=x.shape[1];serum=meta.serum_percent.to_numpy(float);known=serum[np.isfinite(serum)];self.serum_median=float(np.median(known)) if len(known) else 0.;continuous=np.column_stack([x,np.where(np.isfinite(serum),serum,self.serum_median)]);self.scaler=StandardScaler().fit(continuous);self.frequent={};self.category_lists=[]
        for field in CAT_FIELDS:
            values=meta[field].astype(str);counts=values.value_counts();kept=sorted(str(k) for k,v in counts.items() if k not in [MISSING,OTHER] and v>=MIN_COUNTS[field]);self.frequent[field]=set(kept);self.category_lists.append(kept+[MISSING,OTHER])
        mapped=self.map_categories(meta);self.encoder=OneHotEncoder(categories=self.category_lists,handle_unknown='ignore',sparse_output=False,dtype=np.float64).fit(mapped);raw=self.unmixed(x,meta);self.center=raw.mean(axis=0);self.feature_names=[f'gene:{j}' for j in range(self.n_genes)]+['culture:serum_percent','culture:serum_missing']+self.encoder.get_feature_names_out(CAT_FIELDS).tolist();assert raw.shape[1]==len(self.feature_names);self.train_n=len(x);return self

    def map_categories(self,meta):
        return np.column_stack([[str(v) if str(v)==MISSING or str(v) in self.frequent[field] else OTHER for v in meta[field]] for field in CAT_FIELDS])

    def unmixed(self,x,meta):
        x=np.asarray(x,float);assert x.shape[1]==self.n_genes and len(x)==len(meta);serum=meta.serum_percent.to_numpy(float);missing=~np.isfinite(serum);continuous=np.column_stack([x,np.where(missing,self.serum_median,serum)]);z=self.scaler.transform(continuous);return np.column_stack([z,missing.astype(float),self.encoder.transform(self.map_categories(meta))])

    def transform(self,x,meta):
        z=self.unmixed(x,meta)-self.center;assert np.isfinite(z).all();return z

    def diagnostic(self,meta):
        unknown={field:int(sum(str(v)!=MISSING and str(v) not in self.frequent[field] for v in meta[field])) for field in CAT_FIELDS}
        return dict(train_n=self.train_n,features=len(self.feature_names),gene_features=self.n_genes,serum_median=self.serum_median,known_category_counts={f:len(v) for f,v in self.frequent.items()},other_or_unseen_rows=unknown,missing_medium_rows=int(np.sum(meta.medium_code==MISSING)),missing_serum_rows=int(meta.serum_missing.sum()))
