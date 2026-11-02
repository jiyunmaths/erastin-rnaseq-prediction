from pathlib import Path
import shutil
source=Path('outputs/erastin_five_gene_comparison');out=Path('outputs/erastin_elastic_net_comparison')
for name in ['model_helpers.py','weighted_mlp.py','metric_helpers.py','evaluate.py','make_splits.py']:shutil.copyfile(source/name,out/name)
