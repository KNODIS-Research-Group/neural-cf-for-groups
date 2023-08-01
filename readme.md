# Recommending items to group of users using Neural Collaborative Filtering

This work has been published in [Neural Computing & Applications](https://www.springer.com/journal/521) jornal on 22 March 2023. The publication can be found in this [link](https://link.springer.com/article/10.1007/s00521-023-08410-6), DOI [10.1007/s00521-023-08410-6](https://doi.org/10.1007/s00521-023-08410-6), and preprint version [here](https://arxiv.org/abs/2303.07001).

This project contains the following directories:

- `java`: it contains the `ExportTrainTestSplit` class used to download the train/test splits of the datasets provided by CF4J and the `SampleGroups` class used to random generate groups of users.
- `data`: it cointains the data exported from Java to be used by pyhton. MyAnimeList dataset files are not incluided in the repo due to size limitations. They can be generated using the Java classes within the fixed random seed.
- `python`: it contains all code for training and evaluation of models.


Cite:
```
@article{10.1007/s00521-023-08410-6,
  author = {Due\~{n}as-Ler\'{\i}n, Jorge and Lara-Cabrera, Ra\'{u}l and Ortega, Fernando and Bobadilla, Jes\'{u}s},
  title = {Neural Group Recommendation Based on a Probabilistic Semantic Aggregation},
  year = {2023},
  issue_date = {Jul 2023},
  publisher = {Springer-Verlag},
  address = {Berlin, Heidelberg},
  volume = {35},
  number = {19},
  issn = {0941-0643},
  url = {https://doi.org/10.1007/s00521-023-08410-6},
  doi = {10.1007/s00521-023-08410-6},
  journal = {Neural Comput. Appl.},
  month = {mar},
  pages = {14081–14092},
  numpages = {12},
  keywords = {Deep learning, Group recommender system, Aggregation models, Collaborative filtering}
}
```

## Python layout

```txt
- data
  \- dataset1\
  \- dataset2\
  ...
- models
  \- ... h5
- notebooks
  \- researchs in notebooks
- results
  \- evaluation result files
- src
  \- data
    \- rs-data-python (lib)
  \- eval
  \- models
  \- train
  \- utils
```

### Execution

Please set yout PYTHONPATH variable to include this directory

```
PYTHONPATH="/path-to-this-folder/src/data/rs-data-python"
export PYTHONPATH
```

Then you can import all datasets

```
$ python

Python 3.8.5 | packaged by conda-forge | (default, Jun 24 2021, 16:55:52)
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import data_datasets
>>> dataset = data_datasets.DatasetML1M()
>>> print(dataset.info())
DS: ml1m
Train rows: 900188
Val rows: 0
Test rows: 100021
Users: 6040
Items: 3706
Ratings: 1 - 5
>>>
```


#### Train

Generate models ```.h5``` in ```models``` directory.

```python
python src/train/train-all.py --dataset ml1m
python src/train/train-all.py --dataset ft
python src/train/train-all.py --dataset anime
```

#### Evaluate

Generate results in ```results``` directory.

```python
python src/eval/eval-agg-individual-model-all.py
```

#### Visualization

See ```notebooks\tables.ipynb``` to generate result tables.
