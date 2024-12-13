import torch
import torch.nn as nn
from torch.nn.functional import relu


class EEGtoMEGUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder 
        # First Level
        self.e11 = nn.Conv1d(77, 128, kernel_size=3, padding='same')    
        self.e12 = nn.Conv1d(128, 128, kernel_size=3, padding='same')   
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)         
        # [74, 250] → [128, 250] → [128, 125]

        # Second Level
        self.e21 = nn.Conv1d(128, 256, kernel_size=3, padding='same')   
        self.e22 = nn.Conv1d(256, 256, kernel_size=3, padding='same')   
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) 
        # [128, 125] → [256, 125] → [256, 62]

        # Third Level
        self.e31 = nn.Conv1d(256, 512, kernel_size=3, padding='same')   
        self.e32 = nn.Conv1d(512, 512, kernel_size=3, padding='same')   
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)        
        # [256, 62] → [512, 62] → [512, 31]

        # Bridge
        self.bridge1 = nn.Conv1d(512, 1024, kernel_size=3, padding='same')
        self.bridge2 = nn.Conv1d(1024, 1024, kernel_size=3, padding='same')

        # Decoder (modified)
        self.upconv1 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.d11 = nn.Conv1d(1024, 512, kernel_size=3, padding='same')   
        self.d12 = nn.Conv1d(512, 512, kernel_size=3, padding='same')    

        self.upconv2 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2, padding=0)
        self.d21 = nn.Conv1d(512, 256, kernel_size=3, padding='same')    
        self.d22 = nn.Conv1d(256, 256, kernel_size=3, padding='same')    

        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2, padding=0)
        self.d31 = nn.Conv1d(256, 128, kernel_size=3, padding='same')    
        self.d32 = nn.Conv1d(128, 128, kernel_size=3, padding='same')    

        # Output layer (same as before)
        self.outconv = nn.Conv1d(128, 102, kernel_size=1)           
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Save input size for later
        input_size = x.size(-1)
        
        # Encoder
        xe11 = self.relu(self.e11(x))
        xe12 = self.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)
        xp1 = self.dropout(xp1)

        xe21 = self.relu(self.e21(xp1))
        xe22 = self.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)
        xp2 = self.dropout(xp2)

        xe31 = self.relu(self.e31(xp2))
        xe32 = self.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)
        xp3 = self.dropout(xp3)

        # Bridge
        xb1 = self.relu(self.bridge1(xp3))
        xb2 = self.relu(self.bridge2(xb1))
        xb2 = self.dropout(xb2)

        # Decoder with size matching
        xu1 = self.upconv1(xb2)
        xu1 = torch.nn.functional.pad(xu1, [0, xe32.size(-1) - xu1.size(-1)])  # Pad if necessary
        xu11 = torch.cat([xu1, xe32], dim=1)
        xd11 = self.relu(self.d11(xu11))
        xd12 = self.relu(self.d12(xd11))
        xd12 = self.dropout(xd12)

        xu2 = self.upconv2(xd12)
        xu2 = torch.nn.functional.pad(xu2, [0, xe22.size(-1) - xu2.size(-1)])  # Pad if necessary
        xu22 = torch.cat([xu2, xe22], dim=1)
        xd21 = self.relu(self.d21(xu22))
        xd22 = self.relu(self.d22(xd21))
        xd22 = self.dropout(xd22)

        xu3 = self.upconv3(xd22)
        xu3 = torch.nn.functional.pad(xu3, [0, xe12.size(-1) - xu3.size(-1)])  # Pad if necessary
        xu33 = torch.cat([xu3, xe12], dim=1)
        xd31 = self.relu(self.d31(xu33))
        xd32 = self.relu(self.d32(xd31))

        # Output layer
        out = self.outconv(xd32)
        return out