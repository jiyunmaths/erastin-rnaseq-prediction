from pathlib import Path
import ast,shutil,json,hashlib
out=Path('outputs/erastin_nn_moderate_weights')
shutil.copyfile('outputs/erastin_lineage_weighted/weighted_mlp.py',out/'weighted_mlp.py')
def functions(path,names):
 src=Path(path).read_text();tree=ast.parse(src);nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 return '\n\n'.join(ast.get_source_segment(src,nodes[n]) for n in names)
parts=['"""Evaluation routines copied from the verified earlier comparisons; see helper_sources.json."""\nimport json\nimport numpy as np\nimport pandas as pd\nfrom scipy.stats import spearmanr,rankdata\nMETRICS=["spearman","rmse","r2"]\n']
sources=[('outputs/erastin_model_comparison/compare_models.py',['metrics']),('outputs/erastin_repeated_cv/repeat_cv.py',['weighted_ranker','repeated_group_bootstrap','validate_weighted_bootstrap']),('outputs/erastin_bayesian_comparison/review_bayesian.py',['bootstrap_macro','paired_comparisons','score_predictions'])]
for path,names in sources:parts.append(functions(path,names).replace('prior.metrics','metrics'))
(out/'evaluation.py').write_text('\n\n'.join(parts)+'\n')
(out/'helper_sources.json').write_text(json.dumps(dict(sources=[dict(path=p,functions=ns,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p,ns in sources],substitutions=['score_predictions: prior.metrics -> local copied metrics']),indent=2)+'\n')
