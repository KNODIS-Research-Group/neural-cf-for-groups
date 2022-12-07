# rs-data-python
RS data python

## System wide available

Please set yout PYTHONPATH variable to include this directory

```
PYTHONPATH="/workspace/rs-data-python"
export PYTHONPATH
```

Then you can import all datasets

```
$ python

Python 3.8.5 | packaged by conda-forge | (default, Sep 24 2020, 16:55:52) 
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import data_movielens
>>> dataset = data_movielens.RS100K()
Number of users: 610, Number of Movies: 9724, Min rating: 0.5, Max rating: 5.0
Max user id 609 Max item id 9723
100836
>>> 
# Datasets for RS experiments

TODO: Make a python package

