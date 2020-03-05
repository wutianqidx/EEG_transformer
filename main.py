import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader

from data import EEGDataset, collate_wrapper
from model import EEGtoReport, EEGTransformer, loss_func

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size (default=32)")
    args = parser.parse_args()
    args.learning_rate = 0.001
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = 1337
    torch.manual_seed(args.seed)
    return args

def run_training(args, dataset, train_loader):
    model = EEGtoReport()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    model = model.to(args.device)
    model.train()
    for epoch in range(args.epochs):
        for batch_ndx, (input, target, len) in enumerate(train_loader):
            print("input", input)
            print("target", target)
            print("len", len)
            output = model(input, target, len)
            # loss = loss_func(output, target)
            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()


def main(args):
    dataset = EEGDataset("eeg_text.pkl")
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_wrapper)
    run_training(args, dataset, train_loader)

if __name__ == '__main__':
    args = get_args()
    main(args)
