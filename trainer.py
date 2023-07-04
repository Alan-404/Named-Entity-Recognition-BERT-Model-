import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.bert import BERT
from model.loss import BERTLoss
from typing import Callable
import os
from model.metric import BERTMetric

class BERTTRainer:
    def __init__(self,
                 token_size: int,
                 num_entities: int,
                 n: int = 6,
                 d_model=512,
                 heads: int = 8,
                 d_ff: int = 2048,
                 dropout_rate: float = 0.1,
                 eps: float = 0.02,
                 activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
                 device: str = 'cpu',
                 checkpoint: str = None) -> None:
        self.model = BERT(token_size, num_entities, n, d_model, heads, d_ff, dropout_rate, eps, activation)
        self.optimizer = optim.Adam(params=self.model.parameters())

        self.loss_func = BERTLoss()
        self.metric_func = BERTMetric()

        self.epoch = 0
        self.loss = 0.0
        self.loss_epoch = 0.0

        self.metric = 0.0
        self.metric_epoch = 0.0

        self.device = device
        self.checkpoint = checkpoint


        self.model.to(self.device)
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch
        }, path)

    def load_model(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']

    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor):
        self.optimizer.zero_grad()      
        outputs = self.model(inputs)

        loss = self.loss_func(outputs, labels)
        loss.backward()
        self.optimizer.step()

        self.loss += loss.item()
        self.loss_epoch += loss.item()

        _, predicted = torch.max(outputs, dim=-1)
        
        score = self.metric_func.score(predicted, labels)
        
        self.metric += score
        self.metric_epoch += score
        
    def build_dataset(self, inputs: torch.Tensor, labels: torch.Tensor, batch_size: int):
        return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=True)
    
    def fit(self, inputs: torch.Tensor, labels: torch.Tensor, epochs: int = 1, batch_size: int=1, mini_batch: int = 1, learning_rate: float = 1e-6):
        dataloader = self.build_dataset(inputs, labels, batch_size)
        num_batches = len(dataloader)

        for params in self.optimizer.param_groups:
            params['lr'] = learning_rate

        self.model.train()

        for _ in range(epochs):
            count = 0
            for index, data in enumerate(dataloader):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                self.train_step(inputs, labels)
                count += 1

                if index%mini_batch == mini_batch-1 or index == num_batches-1:
                    print(f"Epoch: {self.epoch + 1} Batch: {index + 1} Loss: {(self.loss/count):.4f} Accuracy: {(self.metric/count):.4f}")
                    count = 0
                    self.loss = 0.0
                    self.metric = 0.0
            print(f"Epoch: {self.epoch + 1} Train Loss: {(self.loss_epoch/num_batches):.4f} Accuracy: {(self.metric_epoch/num_batches):.4f}")
            self.epoch += 1
            self.loss_epoch = 0.0
            self.metric_epoch = 0.0

        if self.checkpoint is not None:
            self.save_model(self.checkpoint)
        else:
            self.save_model("./model.pt")