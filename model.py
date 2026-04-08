import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedBrainTumorCNN(nn.Module):
    def __init__(self):
        super(EnhancedBrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # To be adjusted dynamically
        self.fc2 = nn.Linear(512, 2)  # Output layer for binary classification
        self.dropout = nn.Dropout(0.5)

        # Adaptive pooling to handle varying input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))  # Force output to 16x16

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Adaptive pooling to ensure consistent size
        x = self.adaptive_pool(x)
        
        # Flatten the output dynamically
        x = x.view(x.size(0), -1)  # Flattening the output for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model1(model_path='Models/best_model.pth', device=None):
    model = EnhancedBrainTumorCNN()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Move model to the appropriate device (CPU/GPU)
    model.eval()  # Set the model to evaluation mode
    return model