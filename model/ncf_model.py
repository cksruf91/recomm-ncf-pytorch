from collections import OrderedDict
from typing import Iterable

import torch
from torch import nn, Tensor

from model.model_util import TorchInterfaceRecomm


class MLP(nn.Module):

    def __init__(self, user_size, item_size, layers=None, dropout=0.2):
        super().__init__()
        if layers is None:
            layers = [64, 32, 16, 8]
        self.layers = layers
        self.dropout = dropout
        
        n_factor = int(self.layers[0]/2)
        self.user_embedding = nn.Embedding(user_size, n_factor)
        self.item_embedding = nn.Embedding(item_size, n_factor)
        
        self.multi_layer = nn.Sequential(
            OrderedDict(
                {f'layer_{i:02d}': self.dense_layer(self.layers[i - 1], self.layers[i]) for i in range(1, len(self.layers))}
            )
        )

    def dense_layer(self, in_dim, out_dim):
        return nn.Sequential(
            # nn.Dropout(p=self.dropout),
            nn.Linear(in_dim, out_dim), 
            nn.ReLU()
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        x = torch.concat([user_emb, item_emb], axis=-1)

        return self.multi_layer(x)


class GMF(nn.Module):

    def __init__(self, user_size, item_size, n_factor):
        super().__init__()
        self.user_embedding = nn.Embedding(user_size, n_factor)
        self.item_embedding = nn.Embedding(item_size, n_factor)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)

        return user_emb * item_emb


class NeuralMF(TorchInterfaceRecomm):
    """https://github.com/hexiangnan/neural_collaborative_filtering"""

    def __init__(self, user_size: int, item_size: int, n_factor: int, layers=None, dropout: float = 0.2,
                 component: list = None, k: int = 10, device: torch.device = torch.device('cpu')):
        super().__init__()
        if layers is None:
            layers = [64, 32, 16, 8]
        self.component = component

        if self.component is None:
            self.component = ['gmf', 'mlp']

        assert len(self.component) > 0

        self.backbone_models = {}
        if 'gmf' in self.component:
            self.gmf = GMF(user_size, item_size, n_factor)
            self.backbone_models['gmf'] = self.gmf

        if 'mlp' in self.component:
            self.mlp = MLP(user_size, item_size, layers, dropout)
            self.backbone_models['mlp'] = self.mlp

        self.output_layer = nn.Linear(int(n_factor * len(self.component)), 1)
        self.k = k
        self.to(device)
        self._init_weight_()

    def _init_weight_(self):
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

            if name == 'output_layer':
                nn.init.kaiming_uniform_(module.weight, a=1, nonlinearity="sigmoid")
        
    def forward(self, user, item):
        vectors = []
        for k in self.component:
            vectors.append(self.backbone_models[k](user, item))

        x = torch.concat(vectors, dim=-1)

        return self.output_layer(x).reshape(-1)

    def _compute_loss(self, data: Iterable, loss_func, optimizer=None, scheduler=None, train=True) -> tuple[
        Tensor, Tensor, Tensor]:
        users, items, labels = data

        pred = self.forward(users, items)
        loss = loss_func(pred, labels)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            self.zero_grad()

        return loss, labels, pred

    def _validation(self, test_dataloader, loss_func) -> tuple[float, list[list], list[list]]:
        self.eval()
        total_step = len(test_dataloader)
        val_loss = 0
        output, label = [], []

        with torch.no_grad():
            for step, data in enumerate(test_dataloader):
                user, items, positive_item = [d.reshape(-1) for d in data]
                # data = [user, item, positive_item]
                loss, _, y_hat = self._compute_loss([user, items, positive_item], loss_func, train=False)
                val_loss += loss.item()

                _, indices = torch.topk(y_hat, k=self.k)
                output.append(list(items[indices].cpu().numpy()))  # model predictions
                label.append([items[0].cpu().item()])  # positive item

        val_loss = val_loss / total_step
        return val_loss, output, label
