import os

import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from common.dataloader import to_sparse_matrix, Iterator, TestIterator
from config import CONFIG
from model.metrics import nDCG
from model.ncf_model import NeuralMF

if __name__ == '__main__':
    save_dir = os.path.join(CONFIG.DATA, '1M')
    train_data = pd.read_csv(os.path.join(save_dir, 'train.tsv'), sep='\t')
    item_meta = pd.read_csv(os.path.join(save_dir, 'item_meta.tsv'), sep='\t')
    user_meta = pd.read_csv(os.path.join(save_dir, 'user_meta.tsv'), sep='\t')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    matrix = to_sparse_matrix(train_data, 'user_id', 'item_id', 'Rating')
    train_iterator = Iterator(matrix, 4, device=device)
    train_dataloader = DataLoader(train_iterator, batch_size=256, shuffle=True)

    test_iterator = TestIterator(os.path.join(save_dir, 'negative_test.dat'))
    test_dataloader = DataLoader(test_iterator, batch_size=1, shuffle=False)

    n_user, n_item = matrix.shape
    model = NeuralMF(n_user, n_item, hidden_size=8, layers=[64, 32, 16, 8], component=['gmf', 'mlp'])

    cross_entropy = BCEWithLogitsLoss()
    optim = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    metrics = [nDCG()]

    model.fit(10, train_dataloader, test_dataloader, loss_func=cross_entropy, optimizer=optim, metrics=metrics)
