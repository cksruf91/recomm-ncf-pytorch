import argparse
import os

import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from common.dataloader import to_sparse_matrix, Iterator, TestIterator
from config import CONFIG
from model.callbacks import ModelCheckPoint, MlflowLogger
from model.metrics import nDCG, RecallAtK
from model.ncf_model import NeuralMF


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='10M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    parser.add_argument('-v', '--model_version', required=True, help='모델 버전', type=str)
    parser.add_argument('-k', '--eval_k', default=10, help='', type=int)
    parser.add_argument('-f', '--factor', default=8, help='embedding size', type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, help='learning rate', type=float)
    parser.add_argument('-bs', '--batch_size', default=256, help='batch size', type=int)
    parser.add_argument('-ns', '--negative_size', default=4, help='train negative sample size', type=int)
    parser.add_argument('-ly', '--layers', default=[64, 32, 16, 8], help='mlp layer size', type=int, nargs='+')

    return parser.parse_args()


if __name__ == '__main__':
    argument = args()

    save_dir = os.path.join(CONFIG.DATA, argument.dataset)
    train_data = pd.read_csv(os.path.join(save_dir, 'train.tsv'), sep='\t')
    item_meta = pd.read_csv(os.path.join(save_dir, 'item_meta.tsv'), sep='\t')
    user_meta = pd.read_csv(os.path.join(save_dir, 'user_meta.tsv'), sep='\t')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    matrix = to_sparse_matrix(train_data, user_meta.user_id.max() + 1, item_meta.item_id.max() + 1, 'user_id',
                              'item_id', 'Rating')
    train_iterator = Iterator(matrix, argument.negative_size, device=device)
    train_dataloader = DataLoader(train_iterator, batch_size=argument.batch_size, shuffle=True)

    test_iterator = TestIterator(os.path.join(save_dir, 'negative_test.dat'))
    test_dataloader = DataLoader(test_iterator, batch_size=1, shuffle=False)

    n_user, n_item = matrix.shape
    model = NeuralMF(n_user, n_item, hidden_size=argument.factor, layers=argument.layers, component=['gmf', 'mlp'],
                     device=device)

    cross_entropy = BCEWithLogitsLoss()
    optim = Adam(model.parameters(), lr=argument.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                 amsgrad=False)

    model_params = {
        'Factor': argument.factor,  # 500
        'learningRate': argument.learning_rate,
        'loss': 'BCEWithLogitsLoss',
        'optimizer': 'Adam',
        'k': argument.eval_k,
        'batchSize': argument.batch_size,
        'negative_size': argument.negative_size,
        'layers': argument.layers,
        'num_users': n_user, 'num_items': n_item
    }

    metrics = [nDCG(), RecallAtK()]
    model_version = f'nmf_v{argument.model_version}'
    callback = [
        ModelCheckPoint(os.path.join('result', model_version + '_loss{val_loss:1.3f}_nDCG{val_nDCG:1.3f}')),
        MlflowLogger(experiment_name='1M', model_params=model_params, run_name=model_version, log_model=True)
    ]

    print(f"device : {device}")
    model.fit(50, train_dataloader, test_dataloader, loss_func=cross_entropy, optimizer=optim, metrics=metrics,
              callback=callback)
