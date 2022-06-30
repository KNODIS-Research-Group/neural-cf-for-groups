# Recommending items to group of users using Neural Collaborative Filtering

This work has been submitted to Neural Computing & Applications jornal on ######.

This project contains the following directories:

- `java`: it contains the `ExportTrainTestSplit` class used to download the train/test splits of the datasets provided by CF4J and the `SampleGroups` class used to random generate groups of users.
- `data`: it cointains the data exported from Java to be used by pyhton. MyAnimeList dataset files are not incluided in the repo due to size limitations. They can be generated using the Java classes within the fixed random seed.
- `python`:
  - `rs-data-python`: Datasets processing and transformation
  - `dnn-cf-groups`: Models and evaluations

## rs-data-python

Please set yout PYTHONPATH variable to include this directory

```
PYTHONPATH="/path-to-this-folder/rs-data-python"
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


## dnn-cf-groups

### Train

To train all models run ```ncf-groups-train-all.py``` with the output dir where models will be saved and the dataset.
```
python ncf-groups-train-all.py
ncf-groups-train-all.py [-h] --outdir OUTDIR --dataset DATASET [--embacti EMBACTI]
```

Example:
```
python ncf-groups-train-all.py --outdir results --dataset data_groups.GroupDataML1M
```

### Evaluation
To evaluate the models with groups information, you need to set up the script variable ```DS``` and run ```ncf-groups-eval-all.sh```

### Results

You can fin the code to generate the results in:

``` ncf-groups-graph.ipynb ```
