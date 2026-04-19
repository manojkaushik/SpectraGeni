import torch

# Parameters
INPUT_DIM = 943
LATENT_DIM = 64 # Using the larger latent dim from your VAE_utils
LR = 1e-4 # A slightly lower learning rate is often more stable
EPOCHS = 5000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 2000
NUM_CLASSES = 6