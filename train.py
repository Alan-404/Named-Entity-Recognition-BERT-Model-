import torch
from trainer import BERTTRainer
from argparse import ArgumentParser
import json
from preprocessing.data import DataProcessor
import pickle
def load_data(path: str):
    with open(path, 'rb') as file:
        return pickle.load(file)
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--activation", type=str, default='gelu')
    parser.add_argument("--device", type=str, default='cpu')

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--mini_batch", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    

    args = parser.parse_args()

    if args.activation == 'gelu':
        args.activation = torch.nn.functional.gelu
    else:
        args.activation = torch.nn.functional.relu

    processor = DataProcessor(args.tokenizer_path)

    trainer = BERTTRainer(
                        token_size=len(processor.dictionary),
                        num_entities=len(processor.entities),
                        n=args.n,
                        d_model=args.d_model,
                        heads=args.heads,
                        d_ff=args.d_ff,
                        dropout_rate=args.dropout_rate,
                        eps=args.eps,
                        activation=args.activation,
                        device=args.device,
                        checkpoint=args.checkpoint
                    )
    inputs, labels = processor.load_data(args.data_folder)
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)

    trainer.fit(
        inputs,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mini_batch=args.mini_batch,
        learning_rate=args.learning_rate
    )