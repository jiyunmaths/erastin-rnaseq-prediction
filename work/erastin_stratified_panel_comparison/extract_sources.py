from pathlib import Path
import ast,shutil,json,hashlib
out=Path('outputs/erastin_stratified_panel_comparison');prior=Path('outputs/erastin_lineage_stratified_nn')
for name in ['model_helpers.py','weighted_mlp.py','metric_helpers.py','evaluate.py','make_splits.py']:shutil.copyfile(prior/name,out/name)
def extract(path,names):
 s=Path(path).read_text();nodes={n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
 return '\n\n'.join(ast.get_source_segment(s,nodes[name]) for name in names)
source='outputs/erastin_model_comparison/compare_models.py'
(out/'ridge_helpers.py').write_text('"""Unchanged ridge path from the previous panel comparison."""\nimport numpy as np\nfrom scipy.linalg import svd\nfrom sklearn.preprocessing import StandardScaler,OneHotEncoder\n\n'+extract(source,['prepare_features','ridge_path'])+'\n')
source='outputs/erastin_xgboost_comparison/run_comparison.py';s=Path(source).read_text();nodes=ast.parse(s).body;names=['STRUCTURES','ROUNDS','GRID','FIXED'];parts=[]
for node in nodes:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:parts.append(ast.get_source_segment(s,node))
(out/'xgb_helpers.py').write_text('"""Unchanged XGBoost settings from the verified earlier comparison."""\nimport itertools\n\n'+'\n'.join(parts)+'\n\n'+extract(source,['parameters'])+'\n')
