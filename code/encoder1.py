import torch
import torch.nn as nn


# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(x)

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
# Training class
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for images, _ in self.train_loader:
                images = images.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, images)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(self.train_loader):.4f}")
        
        print("Training complete. Model saved.")

# Testing class
class Tester:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def test_model(self):
        self.model.eval()
        with torch.no_grad():
            images, _ = next(iter(self.test_loader))
            images = images.to(self.device)
            reconstructed = self.model(images)
            images = images.cpu().numpy()
            reconstructed = reconstructed.cpu().numpy()
        
        fig, axes = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            axes[0, i].imshow(np.transpose(images[i], (1, 2, 0)) * 0.5 + 0.5)
            axes[0, i].axis('off')
            axes[1, i].imshow(np.transpose(reconstructed[i], (1, 2, 0)) * 0.5 + 0.5)
            axes[1, i].axis('off')
        plt.savefig("reconstructed_images.png")
        plt.close()
        print("Reconstructed images saved.")

    def plot_tsne(self):
        self.model.eval()
        latent_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                latent_vector = self.model.encoder(images)
                latent_list.append(latent_vector.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
        latent_vectors = np.concatenate(latent_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        tsne = TSNE(n_components=2, random_state=0)
        latent_tsne = tsne.fit_transform(latent_vectors)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)
        plt.colorbar(scatter)
        plt.title('t-SNE of Latent Space')
        plt.savefig('latent_tsne.png')
        plt.close()
        print("t-SNE plot saved.")
