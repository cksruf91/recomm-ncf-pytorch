# recomm-ncf-pytorch

___work in progress___
## Dataset
* [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/)
```bash
cd {project Dir}/recomm-hrnn-pytorch/datasets/movielens
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

* [MovieLens 10M Dataset](https://grouplens.org/datasets/movielens/10m/)
```bash
cd {project Dir}/recomm-hrnn-pytorch/datasets/movielens
wget https://files.grouplens.org/datasets/movielens/ml-10m.zip
unzip ml-10m.zip
```

## data preprocess
```shell
python preprocess.py -d 1M
```

## train model
```shell
python ncf_train.py -d 1M -v 0.1.0 -k 10 -f 8 -lr 0.001 -bs 256 -ns 4 -ly 64 32 16 8
```

## Inference model
```shell
python inference.py -d 1M -w nmf_v0.1.0e14_loss0.157_nDCG0.359.zip -k 10 --user 4452
```
