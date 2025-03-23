import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse

import encoder1 as en
import torch.nn as nn
import torch.optim as optim

NUM_CLASSES = 10

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, help='Size of each batch')
    parser.add_argument('--latent-dim', default=128, type=int, help='encoding dimension')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--self-supervised', action='store_true', default=False,
                        help='Whether train self-supervised with reconstruction objective, or jointly with classifier for classification objective.')
    return parser.parse_args()
    

if __name__ == "__main__":

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  #one possible convenient normalization. You don't have to use it.
    ])

    args = get_args()
    freeze_seeds(args.seed)
                
                                           
    if args.mnist:
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=False, transform=transform)
    else:
        train_dataset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
        
    # When you create your dataloader you should split train_dataset or test_dataset to leave some aside for validation



   # ---------- dataloader code --------------

    validation_split = 0.2
    validation_size = int(len(train_dataset) * validation_split)
    train_size = len(train_dataset) - validation_size

    train_subset, val_subset = random_split(train_dataset, [train_size, validation_size])


    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2, 
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )



   # ---------- dataloader code --------------



        # Instantiate model, loss function, and optimizer
    latent_dim = 128
    autoencoder = en.Autoencoder(latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Train the model
    trainer = en.Trainer(autoencoder, train_loader, val_loader, criterion, optimizer, device)
    trainer.train(epochs=20)

    # Save model
    torch.save(autoencoder.state_dict(), "autoencoder_cifar10.pth")

    # Load trained model
    autoencoder.load_state_dict(torch.load("autoencoder_cifar10.pth"))
    autoencoder.to(device)

    tester = en.Tester(autoencoder, test_loader, device)
    tester.test_model()
    tester.plot_tsne()
