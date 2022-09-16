# -*- coding: utf-8 -*-
import argparse
import sys
import os

"""
Read two csv and do a hypothesis testing
"""

import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

text = 'Compare distributions of data.'
parser = argparse.ArgumentParser(description=text)

parser.add_argument('--gfile', type=str, required=True, help="Group file")
parser.add_argument('--s1', type=str, required=True, help="Sampled data 1")
parser.add_argument('--s2', type=str, required=True, help="Sampled data 2")

args = parser.parse_args()

group = pd.read_csv(args.gfile)
group['mean'] = group.filter(regex='^rating',axis=1).mean(axis=1, skipna=True)

d1 = pd.read_csv(args.s1, header=0, names=["s1"])#.round(4)
d2 = pd.read_csv(args.s2, header=0, names=["s2"])#.round(4)

data = pd.concat([group, d1,d2], axis=1)

def infer(stat, p):
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')


def eval_mannwhitneyu(data, index):
    print("mannwhitneyu")
    stat, p = mannwhitneyu(data[index[0]], data[index[1]])
    infer(stat, p)

def eval_wilcoxon(data, index):
    print("wilcoxon")
    stat, p = wilcoxon(data[index[0]], data[index[1]])
    infer(stat, p)

def eval_stat(test_name, data, index):
    print(test_name)
    print(data[index].describe(percentiles=[]))
    eval_mannwhitneyu(data, index)
    eval_wilcoxon(data, index)
    print("-"*40)


"""
Análisis de distribuciones
"""
eval_stat("Distribución de las recomendaciones", data, ['s1','s2'])

"""
Análisis de distribuciones del error respecto media
"""

data['s1_abs_mean'] = abs(data['mean'].subtract(data['s1'],axis=0))
data['s2_abs_mean'] = abs(data['mean'].subtract(data['s2'],axis=0))
eval_stat("Distribución del error absoluto", data, ['s1_abs_mean','s2_abs_mean'])


data['s1_mean'] = data['mean'].subtract(data['s1'],axis=0)
data['s2_mean'] = data['mean'].subtract(data['s2'],axis=0)
eval_stat("Distribución del error", data, ['s1_mean','s2_mean'])
