# recomm-ncf-pytorch

## data preprocess
```shell
python preprocess.py -d 1M
```

## train model
```shell
python ncf_train.py -d 1M -v 0.1.0 -k 10 -f 8 -lr 0.001 -bs 256 -ns 4 -ly 64 32 16 8
```