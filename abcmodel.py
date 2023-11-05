import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants defining the gestational age range
GESTATIONAL_AGE_MIN = 15  # Minimum gestational age in weeks
GESTATIONAL_AGE_MAX = 40  # Maximum gestational age in weeks
NUM_AGES = GESTATIONAL_AGE_MAX - GESTATIONAL_AGE_MIN + 1  # Total number of age classes

class ABCFramework(nn.Module):
    def __init__(self, num_heads, dim_head):
        super(ABCFramework, self).__init__()
        
        # Convolutional Base
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1)

        #Set window and img size
        self.window_size = window_size
        self.img_size = img_size
        
        # Parameters for Q, K, V
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        # Initialize the relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size * window_size, num_heads))
        )
        # Linear transformations for Q, K, V
        self.to_qkv = nn.Linear(9, num_heads * dim_head * 3, bias=False)
        
        # Relative positional encoding
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * 160 - 1) * (2 * 160 - 1), num_heads))
        
        # Additional layers to transform attention output into gestational age distribution
        self.fc1 = nn.Linear(dim_head * num_heads, 128)  # Example dimensions
        self.fc2 = nn.Linear(128, NUM_AGES)
    

    def forward(self, x):
        # Convolutional base
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        # Flatten and pass through linear layers to get Q, K, V
        h, w = x.shape[-2:]
        qkv = self.to_qkv(x.flatten(2)).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(-1, self.num_heads, self.dim_head), qkv)
        
        # Compute the relative positional encoding
        relative_position_bias = self.compute_relative_position_bias(h, w)
        
        # Scaled dot-product attention with relative positional encoding
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale + relative_position_bias
        attn = attn_logits.softmax(dim=-1)
        attn_output = (attn @ v).transpose(1, 2).reshape(-1, h, w, self.dim_head * self.num_heads)
        
        # Shifting the window partition
        attn_output = self.shift_window_partition(attn_output)

      # Transform attention output into gestational age distribution
        x = attn_output.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        predicted_ages = F.softmax(x, dim=1)
        
        return predicted_ages
        #return attn_output
    
    def compute_relative_position_bias(self, H, W):
        # Assuming window size (M) is 4, and the number of attention heads is 4
        M = self.window_size  # Window size is 4 as given
        num_heads = 4
        
        # Create a meshgrid to compute relative distances between elements in the window
        grid_i, grid_j = torch.meshgrid(torch.arange(M), torch.arange(M), indexing="ij")
        relative_coords = torch.stack([grid_i - M // 2, grid_j - M // 2], dim=-1)
        
        # Flatten the relative coordinates and add an extra dimension for heads
        relative_coords = relative_coords.view(-1, 2).unsqueeze(0)

        
       # Use self.num_heads instead of hardcoding the number of heads
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            self.window_size, self.window_size, self.num_heads)

        
        # Convert relative position indices to linear indices
        relative_position_index = relative_coords[:, :, 0] * 2 * M + relative_coords[:, :, 1] + M
        
        # Get the biases for the relative positions
        relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)].view(
            M, M, num_heads)
        
        return relative_position_bias.permute(2, 0, 1)  # Shape (num_heads, M, M)

    def shift_window_partition(self, x):
        # Shift the window partition based on the provided configuration
        # Assuming x has shape [batch_size, channels, height, width]
        
        B, C, H, W = x.shape
        M = self.window_size  # Window size is 4 as given
        
        # Check if the height and width are divisible by the window size
        if H % M != 0 or W % M != 0:
            raise ValueError("Feature map dimensions must be divisible by the window size.")
        
        # Number of windows in height and width
        num_windows_h = H // M
        num_windows_w = W // M
        
        # Shift size is half the window size
        shift_size = M // 2
        
        # Perform the window partitioning with shift
        shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        
        # Now you have to perform the window attention on the shifted_x
        # and then reverse the shift for the output
        
        # This is a simplified placeholder and needs to be fully implemented based on your specific requirements
        return shifted_x


# Instantiate the model
num_heads = 4
dim_head = 64
window_size = 4
img_size = 160
model = ABCFramework(num_heads, dim_head, window_size, img_size)
