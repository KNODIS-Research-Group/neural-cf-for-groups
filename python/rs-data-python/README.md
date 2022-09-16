# rs-data-python
RS data python

## System wide available

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

# Datasets for RS experiments
