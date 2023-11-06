# train.py

import torch
from model import ABCFramework, kl_divergence_loss, diversity_loss, combined_loss, true_age_distribution
from torch.utils.data import DataLoader
from some_dataset_module import YourDataset  # Replace with your actual dataset module

from validation import validation

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
    loss = nn.kl_div(y_pred.log(), y_true, reduction='batchmean')
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

def validation(data_loader, model, lambda_diversity):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    
    mse_loss_fn = torch.nn.MSELoss(reduction='sum')  # Initialize MSE loss function
    mae_loss_fn = torch.nn.L1Loss(reduction='sum')  # Initialize MAE loss function

    with torch.no_grad():  # No need to track gradients for validation
        for images, true_ages in data_loader:
            predicted_ages, attn_output = model(images)
            true_age_distributions = true_age_distribution(true_ages, std_dev=2)
            loss = combined_loss(predicted_ages, true_age_distributions, attn_output, lambda_diversity)

            # Calculate MSE and MAE
            mse = mse_loss_fn(predicted_ages, true_age_distributions)
            mae = mae_loss_fn(predicted_ages, true_age_distributions)

            total_loss += loss.item()
            total_mse += mse.item()
            total_mae += mae.item()

    # Compute the average losses
    avg_loss = total_loss / len(data_loader)
    avg_mse = total_mse / len(data_loader)
    avg_mae = total_mae / len(data_loader)

    return avg_loss, avg_mse, avg_mae



# Training Loop
def train(model, train_ds, val_ds, model_path, tensor_path, epochs, lambda_diversity):
    # Sets up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Initialize the logger
    writer = open(tensor_path, 'w')
    writer.write('Epoch, Train_loss, Val_loss, MSE, MAE\n')

    best_loss = float('inf')

    # Trains for all epochs
    for epoch in range(epochs):
        print('-' * 15)
        print(f'epoch {epoch + 1}/{epochs}')
        model.train()  # Set the model to training mode

        total_loss = 0.0

        for images, true_ages in train_ds:
            optimizer.zero_grad()

            # Forward pass through the model
            predicted_ages, attn_output = model(images)

            # Calculate the true age distributions
            true_age_distributions = true_age_distribution(true_ages, std_dev=2)

            # Calculate the combined loss
            loss = combined_loss(predicted_ages, true_age_distributions, attn_output, lambda_diversity)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_ds)

        # Validation
        avg_val_loss, avg_mse, avg_mae = validation(val_ds, model, lambda_diversity)

        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')

        writer.write(f"{epoch + 1}, {avg_train_loss}, {avg_val_loss}, {avg_mse}, {avg_mae}\n")

        # Save the model checkpoint
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, f'{model_path}/model_best.pth')
            print(f'Saved best model in epoch: {epoch + 1}')

    writer.close()

# Main Function
def main():
    # Model parameters
    num_heads = 4
    dim_head = 64
    window_size = 4
    img_size = 160
    num_ages = NUM_AGES  # Replace with the actual number of ages you have

    # Initialize model
    model = ABCFramework(num_heads, dim_head, window_size, img_size, num_ages)

    # DataLoader, etc.
    dataset = YourDataset(img_size=img_size)  # Ensure your dataset provides images of the correct size
    train_ds = DataLoader(dataset, batch_size=32, shuffle=True)
    val_ds = DataLoader(dataset, batch_size=32, shuffle=False)  # Ideally, this should be a separate validation dataset

    lambda_diversity = 10.0
    epochs = 100

    # Paths for saving models and logs
    model_path = "./model_checkpoints"
    tensor_path = "./logs.csv"

    # Training
    train(model, train_ds, val_ds, model_path, tensor_path, epochs, lambda_diversity)

if __name__ == "__main__":
    main()
