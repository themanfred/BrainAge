# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ABCFramework
from torch.utils.data import DataLoader
from some_dataset_module import YourDataset  # Replace with your actual dataset module

class img_dataset(Dataset):
    def __init__(self, root_dir, view):
        self.root_dir = root_dir
        self.view = view

    def __len__(self):
        if self.view == 'L':
            size = 110
        elif self.view == 'A':
            size = 158
        else:
            size = 126
        return size
    
    def __getitem__(self, idx):
        raw = nib.load(self.root_dir).get_fdata()
        if self.view == 'L':
            n_img = raw[idx,:158,:]    
        elif self.view == 'A':
            n_img = raw[:110,idx,:]
        else:
            n_img = raw[:110,:158,idx]

        num = n_img-np.min(n_img)
        den = np.max(n_img)-np.min(n_img)
        out = np.zeros((n_img.shape[0], n_img.shape[1]))
    
        n_img = np.divide(num, den, out=out, where=den!=0)

        n_img = np.expand_dims(n_img,axis=0)
        n_img = torch.from_numpy(n_img).type(torch.float)

        return n_img
        
# KL Divergence Loss
def kl_divergence_loss(y_pred, y_true):
    loss = F.kl_div(y_pred.log(), y_true, reduction='batchmean')
    return loss

# Diversity Loss
def diversity_loss(attn):
    num_heads = attn.size(1)
    loss = 0.0
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            loss += (attn[:, i] * attn[:, j]).sum()
    return loss

# Combined Loss
def combined_loss(y_pred, y_true, attn, lambda_diversity=10.0):
    kl_loss = kl_divergence_loss(y_pred, y_true)
    div_loss = diversity_loss(attn)
    total_loss = kl_loss + lambda_diversity * div_loss
    return total_loss

# True Age Distribution
def true_age_distribution(true_ages, std_dev=1, min_age=GESTATIONAL_AGE_MIN, max_age=GESTATIONAL_AGE_MAX):
    age_indices = torch.arange(min_age, max_age + 1, dtype=torch.float32).to(true_ages.device)
    true_ages = true_ages.unsqueeze(1)
    var = std_dev ** 2
    distributions = torch.exp(-torch.pow(age_indices - true_ages, 2) / (2 * var))
    distributions /= distributions.sum(dim=1).unsqueeze(1)  # Normalize to sum to 1
    return distributions

# Training Loop
def train(model, data_loader, optimizer, lambda_diversity, window_size, img_size):
    model.train()
    for images, true_ages in data_loader:
        optimizer.zero_grad()
        predicted_ages, attn_output = model(images)  # Model's forward pass now outputs predicted age distributions and attention output
        true_age_distributions = true_age_distribution(true_ages, std_dev=2)
        loss = combined_loss(predicted_ages, true_age_distributions, attn_output, lambda_diversity)
        loss.backward()
        optimizer.step()

# Main Function
def main():
    # Initialize model
    num_heads = 4
    dim_head = 64
    window_size = 4
    img_size = 160
    model = ABCFramework(num_heads, dim_head, window_size, img_size)

    # DataLoader, Optimizer, etc.
    dataset = YourDataset()  # Replace with your actual dataset
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lambda_diversity = 10.0

    # Training
    train(model, data_loader, optimizer, lambda_diversity, window_size, img_size)

if __name__ == "__main__":
    main()
