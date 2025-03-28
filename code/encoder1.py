import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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
        )
    
    def forward(self, x):
        return self.encoder(x)

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
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
    # Define Classifier
class Classifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.encoder[-1].out_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            
        )
    
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        logits = self.classifier(features)
        return logits

# Training class for autoencoder
class AutoencoderTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []

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
            
            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(self.train_loader)}")
        
        print("Autoencoder training complete. Model saved.")

    def plot_loss(self,save_path='autoencoder_training_loss.png'):
            plt.plot(self.train_losses, label='Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Autoencoder Training Loss')
            plt.legend()
            plt.savefig('autoencoder_training_loss.png')
            plt.close

# Training class for classifier
class ClassifierTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.train_accuracies = []

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            corret = 0
            total = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                corret += (predicted == labels).sum().item()
                
            

            avg_train_loss = train_loss / len(self.train_loader)
            avg_train_accuracy = 100* corret / total
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(avg_train_accuracy)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss}")
            print(f"Accuracy: {avg_train_accuracy}")
        
        print("Classifier training complete. Model saved.")

    def plot_loss(self, save_path='classifier_training_loss.png'):
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Classifier Training Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
    
    def plot_accuracy(self, save_path='classifier_training_accuracy.png'):
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Classifier Training Accuracy')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
    
    # Define Classifier (modified to work directly with latent space)
class JointModel(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super(JointModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(encoder.encoder[-1].out_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        logits = self.classifier(latent)
        return logits

# Training class for joint model
class JointModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.train_accuracies = []

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Compute accuracy
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Compute average loss and accuracy
            avg_train_loss = train_loss / len(self.train_loader)
            avg_train_accuracy = 100 * correct / total
            
            # Store metrics
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(avg_train_accuracy)
            
            # Print epoch statistics
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.2f}%")
        
        print("Joint model training complete.")

    def plot_loss(self, save_path='joint_training_loss.png'):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Joint Model Training Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
    
    def plot_accuracy(self, save_path='joint_training_accuracy.png'):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Joint Model Training Accuracy')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

