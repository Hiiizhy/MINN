import torch
import numpy as np
import argparse
import os
from data_loader import dataLoader
from trainer import training

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    same_seeds(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='TaFeng')
    parser.add_argument('--asp', type=int, default=2)
    parser.add_argument('--h', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--nbNUM', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--udim', type=int, default=1)
    parser.add_argument('--isTrain', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--evalEpoch', type=int, default=1)
    parser.add_argument('--testOrder', type=int, default=1)
    config = parser.parse_args()

    if config.isTrain:
        resultFileName = 'all_results_valid'
    else:
        resultFileName = 'all_results'

    print(config)
    dataset = dataLoader(config)

    if config.isTrain:
        config.padIdx = dataset.numItemsTrain
    else:
        config.padIdx = dataset.numItemsTest

    print('start training')
    training(dataset, config, device)
