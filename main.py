import warnings
warnings.filterwarnings("ignore")
import os
import copy
import argparse
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data import EEGDataset, collate_wrapper
from model import EEGtoReport # , loss_func

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=3, help="Batch size (default=32)")
    parser.add_argument('--run', default='', help='Continue training on runX. Eg. --run=run1')
    args = parser.parse_args()
    args.base_output = "checkpoint/"
    args.checkpoint = "eeg_text_checkpoint"
    args.eval_every = 5
    args.learning_rate = 0.0001

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.seed = 1337
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.base_output, exist_ok=True)
    if len(args.run) == 0:
        run_count = len(os.listdir(args.base_output))
        args.run = 'run{}'.format(run_count)
    args.base_output = os.path.join(args.base_output, args.run)
    os.makedirs(args.base_output, exist_ok=True)
    return args

def run_training(args, dataset, train_loader, val_loader):
    checkpoint_path = os.path.join(args.base_output, args.checkpoint)
    dataset_dict = copy.deepcopy(dataset.__dict__)
    del dataset_dict['data']
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = EEGtoReport(eeg_epoch_max=checkpoint['dataset']['max_len'],
                            report_epoch_max=checkpoint['dataset']['max_len_t'],
                            vocab_size=checkpoint['dataset']['vocab_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(
            model.parameters(),
            lr=checkpoint['args']['learning_rate']
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('epochs={},batch_size={},run={},lr={} loss:{}'.format(
        checkpoint['epoch'],checkpoint['args']['batch_size'],checkpoint['args']['run'],
        checkpoint['args']['learning_rate'],checkpoint['loss']))
        epoch_start = checkpoint['epoch']
    else:
        model = EEGtoReport(eeg_epoch_max=dataset.max_len,
                            report_epoch_max=dataset.max_len_t,
                            vocab_size=dataset.vocab_size)
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate
        )
        epoch_start = 0

    model = model.to(args.device)
    #model.train()
    train_metrics = []
    val_metrics = [1000,1000]
    val_epoch = 0 

    for epoch in range(args.epochs):
        model.train()
        train_epoch_metrics = []

        for batch_ndx, (input, target_i, target, length, length_t) in enumerate(train_loader):
            ## print('input:', input.size())
            ## print('target_i:', [x.size() for x in target_i])
            ## print('target:', target.size())
            ## print('length:', length)
            #print('length_t:', length_t)
            output, padded_target = model(input, target_i, length, length_t, args.device)
            #loss = loss_func(output, target, length_t)
            loss = model.loss_func(output, padded_target, length_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_metrics.append(loss.item())
        train_metrics.append(np.mean(train_epoch_metrics))

        if epoch % args.eval_every == args.eval_every-1 or epoch == args.epochs-1:
            torch.save({
            'epoch': epoch_start+epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_metrics[epoch],
            'dataset': dataset_dict,
            'args': vars(args)
            }, checkpoint_path)
            #print('Epoch {}:'.format(args.epochs))

	    # Start Validation
            model.eval() 
            val_epoch_metrics = []
            for batch_ndx, (input, target_i, target, length, length_t) in enumerate(val_loader):
                output, padded_target = model(input, target_i, length, length_t, args.device)
                loss = model.loss_func(output, padded_target, length_t)
                val_epoch_metrics.append(loss.item())

            val_metrics.append(np.mean(val_epoch_metrics))
            # print(val_metrics,epoch)
            #  print(train_metrics[epoch])
            # print(val_metrics[val_epoch+2])

            print('epochs={},run={},lr={} train_loss:{}, val_loss:{}'.format(
                epoch_start+epoch+1,args.run,args.learning_rate, train_metrics[epoch], val_metrics[val_epoch+2]))

            val_epoch = val_epoch+1
            #print(torch.argmax(output,dim=2).view(-1), padded_target)

        # break if starts overfitting with patience 2
        if val_metrics[-1] > val_metrics[-2] and val_metrics[-1] > val_metrics[-3]:
            break

def main(args):
    tic = time.time()
    train_dataset = EEGDataset("dataset/eeg_text_train.pkl")
    val_dataset = EEGDataset("dataset/eeg_text_val.pkl")
    print("train_dataset len:", len(train_dataset))
    print("val_dataset len:", len(val_dataset))
    print('Epoch: ', args.epochs)
    print('batch_size', args.batch_size)
    #print(dataset.max_len_t,dataset.max_len,dataset.vocab_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_wrapper)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_wrapper)
    run_training(args, train_dataset, train_loader, val_loader)
    print('[{:.2f}] Finish training'.format(time.time() - tic))

if __name__ == '__main__':
    """TO DO: train and overfit on one sample at first"""
    args = get_args()
    main(args)
