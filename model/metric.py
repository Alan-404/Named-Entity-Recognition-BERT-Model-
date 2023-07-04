from sklearn.metrics import accuracy_score
import torch
class BERTMetric:
    def __init__(self) -> None:
        pass

    def score(self, outputs: torch.Tensor, labels: torch.Tensor):
        batch_size = outputs.size(0)
        metric = 0.0
        for i in range(batch_size):
            metric += accuracy_score(labels[i].cpu().numpy(), outputs[i].cpu().numpy())

        return metric/batch_size