
#

model_list = [
    'models/ft/gmf_k8_dsft_seed1234.h5',
    'models/ft/mlp_k8_dsft_seed1234.h5',
    'models/ml1m/gmf_k8_dsml1m_seed1234.h5',
    'models/ml1m/mlp_k8_dsml1m_seed1234.h5',
    'models/anime/gmf_k8_dsanime_seed1234.h5',
    'models/anime/mlp_k8_dsanime_seed1234.h5',
]

import sys
import os

for m in model_list:
    os.system(f"python src/eval/eval-agg-individual-model.py --m {m}")
