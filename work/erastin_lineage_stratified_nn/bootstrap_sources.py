from pathlib import Path
import ast,shutil
src=Path('outputs/erastin_nn_moderate_weights');out=Path('outputs/erastin_lineage_stratified_nn')
for name in ['weighted_mlp.py','lineage_weights.py']:shutil.copyfile(src/name,out/name)
def extract(p,names):
 s=p.read_text();nodes={n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
 return '\n\n'.join(ast.get_source_segment(s,nodes[n]) for n in names)
s='"""Unchanged NN fit and serialization helpers from the verified weight comparison."""\nimport hashlib,json,time,warnings\nfrom pathlib import Path\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler\nfrom weighted_mlp import WeightedMLPRegressor\n\n'+extract(src/'run_moderate.py',['sha','clean','dump','fitted_neural'])+'\n'
(out/'model_helpers.py').write_text(s)
s='"""Metric and weighted midrank helpers copied unchanged from the previous comparison."""\nimport numpy as np\nfrom scipy.stats import spearmanr,rankdata\n\n'+extract(src/'evaluation.py',['metrics','weighted_ranker'])+'\n'
(out/'metric_helpers.py').write_text(s)
