#!/bin/bash
echo "Biased .. MLP sin activación media"
python ncf-stat.py --gfile ../rs-data-python/grupos/ml1m/groups-4.csv --s1 mlp_ml1m/ml1m/groups-4-biasedmf-avg.csv --s2 mlp_ml1m/ml1m/groups-4-mlp_ml1m_4_mean_indi.csv

echo "Biased .. MLP sin activación y multionehot"
python ncf-stat.py --gfile ../rs-data-python/grupos/ml1m/groups-4.csv --s1 mlp_ml1m/ml1m/groups-4-biasedmf-avg.csv --s2 mlp_ml1m/ml1m/groups-4-mlp_ml1m_4_trunc.csv 


echo "MLP sin activación media .. MLP sin activación y multionehot"
python ncf-stat.py --gfile ../rs-data-python/grupos/ml1m/groups-4.csv --s1 mlp_ml1m/ml1m/groups-4-mlp_ml1m_4_mean_indi.csv --s2 mlp_ml1m/ml1m/groups-4-mlp_ml1m_4_trunc.csv 

echo "MLP con activación media .. MLP con activación y multionehot"
python ncf-stat.py --gfile ../rs-data-python/grupos/ml1m/groups-4.csv --s1 mlp_ml1m/ml1m/groups-4-mlp_ml1m_relu_10_mean_indi.csv --s2 mlp_ml1m/ml1m/groups-4-mlp_ml1m_relu_10_trunc.csv 

echo "Biased .. MLP con activación y multionehot"
python ncf-stat.py --gfile ../rs-data-python/grupos/ml1m/groups-4.csv --s1 mlp_ml1m/ml1m/groups-4-biasedmf-avg.csv --s2 mlp_ml1m/ml1m/groups-4-mlp_ml1m_relu_10_trunc.csv 