import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants defining the gestational age range
GESTATIONAL_AGE_MIN = 15  # Minimum gestational age in weeks
GESTATIONAL_AGE_MAX = 40  # Maximum gestational age in weeks
NUM_AGES = GESTATIONAL_AGE_MAX - GESTATIONAL_AGE_MIN + 1  # Total number of age classes



class WindowAttention(nn.Module):
    def __init__(self, num_heads, dim_head, window_size):
        super(WindowAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.window_size = window_size

        # Relative positional encoding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

    def forward(self, q, k, v, h, w):
        # Compute the relative positional encoding
        relative_position_bias = self.compute_relative_position_bias(h, w)

        # Scaled dot-product attention with relative positional encoding
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale + relative_position_bias
        attn = attn_logits.softmax(dim=-1)
        attn_output = (attn @ v).transpose(1, 2).reshape(-1, h, w, self.dim_head * self.num_heads)

        # Shift the window partition
        attn_output = self.shift_window_partition(attn_output)
        return attn_output

    def compute_relative_position_bias(self, H, W):
        # Assuming window size (M) is given, and the number of attention heads is specified
        M = self.window_size
        num_heads = self.num_heads
        
        # Create a meshgrid to compute relative distances between elements in the window
        grid_i, grid_j = torch.meshgrid(torch.arange(M), torch.arange(M), indexing="ij")
        relative_coords = torch.stack([grid_i - M // 2, grid_j - M // 2], dim=-1)
        
        # Flatten the relative coordinates and add an extra dimension for heads
        relative_coords = relative_coords.view(-1, 2).unsqueeze(0)
        
        # Convert relative position indices to linear indices
        relative_position_index = relative_coords[:, :, 0] * 2 * M + relative_coords[:, :, 1] + M
        
        # Get the biases for the relative positions
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            M, M, num_heads
        )
        
        return relative_position_bias.permute(2, 0, 1)  # Shape (num_heads, M, M)
        
    def shift_window_partition(self, x, shift=True):
        # Shift the window partition based on the provided configuration
        # Assuming x has shape [batch_size, channels, height, width]
        
        B, C, H, W = x.shape
        M = self.window_size  # Window size is given
        
        # Check if the height and width are divisible by the window size
        if H % M != 0 or W % M != 0:
            raise ValueError("Feature map dimensions must be divisible by the window size.")
        
        shift_size = M // 2  # Shift size is half the window size
        
        if shift:
            # Perform the window partitioning with forward shift
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))
        else:
            # Perform the window partitioning with reverse shift
            shifted_x = torch.roll(x, shifts=(shift_size, shift_size), dims=(2, 3))
        
        return shifted_x


        
class ABCFramework(nn.Module):
    def __init__(self, num_heads, dim_head):
        super(ABCFramework, self).__init__()

        """
        # Convolutional Base
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1)

        """
        # Applies He initialization to the weights of each convolutional layer.
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
        self.conv_layers = nn.Sequential(
            # Layer 1 (1 -> 3 channels)
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # Layer 2 (3 -> 3 channels)
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # Pooling
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 3 (3 -> 6 channels)
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            # Layer 4 (6 -> 6 channels)
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            # Pooling
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 5 (6 -> 9 channels)
            nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
            # Layer 6 (9 -> 9 channels)
            nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
            # Layer 7 (9 -> 9 channels)
            nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
            # Layer 8 (9 -> 9 channels)
            nn.Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU()
        )
        
        self.conv_layers.apply(init_weights)

        
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
        
       # Window attention module
        self.window_attention = WindowAttention(num_heads, dim_head, window_size)
        
        # Additional layers to transform attention output into gestational age distribution
        self.fc1 = nn.Linear(dim_head * num_heads * window_size * window_size, 128)  # Example dimensions
        self.fc2 = nn.Linear(128, NUM_AGES)
    
    
    def forward(self, x):

        x = self.conv_layers(x)
        """
        # Convolutional base
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        """
        # Flatten and pass through linear layers to get Q, K, V
        h, w = x.shape[-2:]
        qkv = self.to_qkv(x.flatten(2)).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(-1, self.num_heads, self.dim_head), qkv)
        
        # Window attention
        attn_output = self.window_attention(q, k, v, h, w)
    
        # Transform attention output into gestational age distribution
        x = attn_output.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        predicted_ages = F.softmax(x, dim=1)
        
        return predicted_ages



 

# Instantiate the model
num_heads = 4
dim_head = 64
window_size = 4
img_size = 160
model = ABCFramework(num_heads, dim_head, window_size, img_size)
