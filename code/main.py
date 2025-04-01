import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse

import os
import encoder1 as en
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched


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

    args = get_args()

    mean = [0.5, 0.5, 0.5] if not args.mnist else [0.5]
    std = [0.5, 0.5, 0.5] if not args.mnist else [0.5]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  #one possible convenient normalization. You don't have to use it.
    ])

    # Set device
    device = torch.device(args.device)
                
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
    latent_dim = args.latent_dim
    dataset = "MNIST" if args.mnist else "CIFAR10"
    print(f"Using {dataset} dataset")
    
    if args.self_supervised:
        if not os.path.exists(f"autoencoder_{dataset}.pth"):
            print("Training autoencoder")
            # Train autoencoder
            autoencoder = en.Autoencoder(latent_dim, args.mnist).to(device)
            ae_criterion = nn.L1Loss()
            ae_lr = 0.0005 if not args.mnist else 0.0001
            ae_epochs = 30
            ae_optimizer = optim.Adam(autoencoder.parameters(), lr=ae_lr)
            ae_trainer = en.AutoencoderTrainer(autoencoder, train_loader, val_loader, ae_criterion, ae_optimizer, device)

            ae_trainer.train(epochs=ae_epochs)

            torch.save(autoencoder.state_dict(), f"autoencoder_{dataset}.pth")
            ae_trainer.plot_loss(save_path=f"autoencoder_training_loss_{dataset}.png")
            ae_trainer.plot_validation_loss(save_path=f"autoencoder_validation_loss_{dataset}.png")
        else:
            print("autoencoder.pth exists. Loading model.")
            autoencoder = en.Autoencoder(latent_dim, args.mnist).to(device)
            autoencoder.load_state_dict(
                torch.load(f"autoencoder_{dataset}.pth", map_location=device)
                )

        if not os.path.exists(f"classifier_{dataset}.pth"):
            print("Training classifier")
            # Train classifier
            classifier = en.Classifier(autoencoder.encoder, num_classes=10).to(device)
            clf_criterion = nn.CrossEntropyLoss()
            clf_lr = 0.005 if not args.mnist else 0.001
            clf_epochs=20
            clf_optimizer = optim.Adam(classifier.parameters(), lr=clf_lr)
            clf_trainer = en.ClassifierTrainer(classifier, train_loader, val_loader, clf_criterion, clf_optimizer, device)

            clf_trainer.train(epochs=clf_epochs)

            torch.save(classifier.state_dict(), f"classifier_{dataset}.pth")
            clf_trainer.plot_loss(save_path=f"classifier_training_loss_{dataset}.png")
            clf_trainer.plot_accuracy(save_path=f"classifier_training_accuracy_{dataset}.png")  
            clf_trainer.plot_validation_accuracy(save_path=f"classifier_validation_accuracy_{dataset}.png")
        else:
            print("classifier.pth exists. Loading model.")
            classifier = en.Classifier(autoencoder.encoder, num_classes=10).to(device)
            classifier.load_state_dict(torch.load(f"classifier_{dataset}.pth"))
                else: # not self-supervised
        model_path = f"joint_classification_model_{dataset}.pth"

        if not os.path.exists(model_path):
            print("Training joint classification model...")
            encoder = en.JointEncoder(latent_dim=latent_dim).to(device)
            joint_model = en.JointModel(encoder, latent_dim=latent_dim, num_classes=10).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(joint_model.parameters(), lr=0.001)
            epochs = 20

            trainer = en.JointModelTrainer(joint_model, train_loader, val_loader, criterion, optimizer, device)
            trainer.train(epochs)

            torch.save(joint_model.state_dict(), model_path)
            trainer.plot_loss(f"joint_training_loss_{dataset}.png")
            trainer.plot_accuracy(f"joint_training_accuracy_{dataset}.png")
        else:
            print("Model already exists. Loading...")
            encoder = en.Encoder(latent_dim=latent_dim).to(device)
            joint_model = en.JointModel(encoder, latent_dim=latent_dim, num_classes=10).to(device)
            joint_model.load_state_dict(torch.load(model_path))