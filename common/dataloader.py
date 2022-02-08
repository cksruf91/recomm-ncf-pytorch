import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

from common.utils import progressbar


def to_sparse_matrix(df, x_col, y_col, v_col):
    total = len(df)

    num_x = df[x_col].nunique()
    num_y = df[y_col].nunique()
    mat = sp.dok_matrix((num_x + 1, num_y + 1), dtype=np.float32)
    for i, (user, item, rating) in enumerate(zip(df[x_col], df[y_col], df[v_col])):
        progressbar(total, i + 1, prefix='to sparse matrix')
        if rating > 0:
            mat[user, item] = 1.0

    return mat


class Iterator(Dataset):

    def __init__(self, mat, n_negative, device=None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.n_negative = n_negative
        self.user_inputs = []
        self.item_inputs = []
        self.labels = []

        self._generate_dataset(mat)

    def _generate_dataset(self, mat):

        num_items = mat.shape[1]
        length = len(mat.keys())
        for i, (user, item) in enumerate(mat.keys()):
            progressbar(length, i + 1, prefix='generate negative samples')
            self.user_inputs.append(user)
            self.item_inputs.append(item)
            self.labels.append(1)

            for _ in range(self.n_negative):
                j = np.random.randint(num_items)
                while mat.get((user, j)):
                    j = np.random.randint(num_items)
                self.user_inputs.append(user)
                self.item_inputs.append(j)
                self.labels.append(0)

    def _to_tensor(self, value, dtype=torch.int64):
        return torch.tensor(value, device=self.device, dtype=dtype)

    def __getitem__(self, index):
        user = self._to_tensor(self.user_inputs[index])
        item = self._to_tensor(self.item_inputs[index])
        label = self._to_tensor(self.labels[index], dtype=torch.float32)
        return user, item, label

    def __len__(self):
        return len(self.labels)


class TestIterator(Dataset):

    def __init__(self, test_file, device=None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.read_file(test_file)

    def read_file(self, test_file):
        self.data = []
        skip = 0
        with open(test_file, 'r') as f:
            for row in f:
                try:
                    row = [int(r) for r in row.split('\t')]
                except ValueError as e:
                    skip += 1
                    continue
                self.data.append(row)
        print(f'skip count : {skip}')

    def _to_tensor(self, value, dtype=torch.int64):
        return torch.tensor(value, device=self.device, dtype=dtype)

    def __getitem__(self, index):
        user = [self.data[index][0]] * 100
        label = [1] + [0] * 99

        user = self._to_tensor(user)
        item = self._to_tensor(self.data[index][1:])
        label = self._to_tensor(label, dtype=torch.float32)
        return user, item, label

    def __len__(self):
        return len(self.data)
