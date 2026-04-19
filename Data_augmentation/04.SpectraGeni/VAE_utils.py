# VAE_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import copy

class ConvolutionalCVAE(nn.Module):
    """
    ## Architecture Upgrade:
    A Convolutional CVAE that uses 1D Convolutions to better capture
    patterns in sequential hyperspectral data.
    """
    def __init__(self, input_dim, latent_dim, num_classes):
        super(ConvolutionalCVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # --- Label Embedding ---
        # A small embedding for the class label
        self.label_embedding = nn.Embedding(num_classes, 16)

        # --- ENCODER ---
        # Uses 1D convolutions to process the spectral data
        self.encoder_conv = nn.Sequential(
            # Input: (batch, 1, 943)
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3), # (batch, 64, 472)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), # (batch, 128, 236)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), # (batch, 256, 118)
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Flattened size from conv layers + size of label embedding
        self.encoder_fc_input_dim = 256 * 118 + 16 
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.encoder_fc_input_dim, 512),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)

        # --- DECODER ---
        # Initial FC layer to prepare for convolutional upsampling
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + 16, 256 * 118), # Combine z and label, then project
            nn.ReLU()
        )

        # Uses 1D transposed convolutions to upsample back to the original shape
        self.decoder_conv = nn.Sequential(
            # Input: (batch, 256, 118)
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1), # (batch, 128, 236)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1), # (batch, 64, 472)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Final layer needs careful padding/stride to hit 943
            nn.ConvTranspose1d(64, 1, kernel_size=7, stride=2, padding=3, output_padding=0), # (batch, 1, 943)
            nn.Tanh() # Scale output to match standardized data
        )

    def encode(self, x, y):
        # Reshape input for convolution: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Pass through conv layers
        h_conv = self.encoder_conv(x)
        h_flat = h_conv.view(h_conv.size(0), -1) # Flatten
        
        # Get label embedding
        y_embedded = self.label_embedding(y)
        
        # CVAE Change: Concatenate flattened data with the embedded class label
        combined_input = torch.cat([h_flat, y_embedded], dim=1)
        
        h_fc = self.encoder_fc(combined_input)
        mu = self.fc_mu(h_fc)
        log_var = torch.clamp(self.fc_log_var(h_fc), min=-10, max=10)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) + 1e-6
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        # CVAE Change: Concatenate the latent vector with the embedded class label
        y_embedded = self.label_embedding(y)
        combined_input = torch.cat([z, y_embedded], dim=1)
        
        # Pass through initial FC layer and reshape for convolutions
        h_fc = self.decoder_fc(combined_input)
        h_reshaped = h_fc.view(h_fc.size(0), 256, 118)
        
        # Pass through transposed conv layers
        recon_conv = self.decoder_conv(h_reshaped)
        
        # Remove the channel dimension to match input shape
        recon = recon_conv.squeeze(1)
        
        # Final adjustment to ensure exact size
        if recon.shape[1] > self.input_dim:
            recon = recon[:, :self.input_dim]
            
        return recon

    def forward(self, x, y):
        mu, log_var = self.encode(x, y)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, y)
        return recon_x, mu, log_var

def vae_loss_function(recon_x, x, mu, log_var, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss

def train_cvae(model, train_loader, val_loader, epochs, lr, device, patience=25):
    """
    Training loop for the CVAE. Includes early stopping.
    This function does not need to be changed for the new model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {model.__class__.__name__}...")
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data, labels)
            beta = min(1.0, epoch / (0.4 * epochs))
            loss, _, _ = vae_loss_function(recon_batch, data, mu, log_var, beta=beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item()/len(data):.4f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for data, labels in vbar:
                data, labels = data.to(device), labels.to(device)
                recon_batch, mu, log_var = model(data, labels)
                loss, _, _ = vae_loss_function(recon_batch, data, mu, log_var, beta=1.0)
                val_loss += loss.item()
                vbar.set_postfix(val_loss=f"{loss.item()/len(data):.4f}")
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Restoring best model.")
            model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("Training finished.")
    return model

def generate_synthetic_conditional(model, scaler, num_samples, label_to_generate, latent_dim, device):
    """
    Generates data for a SPECIFIC class label.
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        labels = torch.tensor([label_to_generate] * num_samples, dtype=torch.long).to(device)
        synthetic_data_scaled = model.decode(z, labels)
    synthetic_data_original = scaler.inverse_transform(synthetic_data_scaled.cpu().numpy())
    return synthetic_data_original
