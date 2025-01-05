import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGtoMEGUNet(nn.Module):
    """
    A simplified U-Net style model for EEG-to-MEG prediction.
    This model is designed for wavelet-transformed input (3 channels).
    """

    def __init__(self):
        super().__init__()
        
        # ------------------------
        # Encoder
        # ------------------------
        # First Level - Now accepts 3 input channels
        self.e11 = nn.Conv2d(3, 32, kernel_size=(1, 3), padding=(0, 1))  # Changed input channels to 3
        self.e12 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Second Level
        self.e21 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        self.e22 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # ------------------------
        # Bridge (recurrent layer)
        # ------------------------
        self.lstm_hidden = 32
        self.lstm = nn.LSTM(
            input_size=64,    # matches channels from xp2
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # Project LSTM hidden dim back to 64 for decoder
        self.lstm_out_projection = nn.Conv1d(self.lstm_hidden, 64, kernel_size=1)

        # ------------------------
        # Decoder
        # ------------------------
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=(1, 2), stride=(1, 2))
        self.d11 = nn.Conv2d(64 + 64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.d12 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(1, 2), stride=(1, 2))
        self.d21 = nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1))
        self.d22 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1))

        # Output layer - outputs 102 MEG channels
        self.outconv = nn.Conv2d(32, 3, kernel_size=(1, 1))  # Changed from 102 to 3
        
        # Final upsampling to match target size
        self.final_upsample = nn.Upsample(size=(1, 275), mode='bilinear', align_corners=False)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the U-Net for wavelet-transformed EEG to MEG prediction.
        Expected input shape: [batch, channels=3, time=275].
        Returns: [batch, meg_channels=102, time=275].
        """
        # Add spatial dimension for 2D convolutions
        x = x.unsqueeze(2)  # [batch, 3, 1, time]

        # ------------------------
        # Encoder
        # ------------------------
        xe11 = self.relu(self.e11(x))     # -> [batch, 32, 1, time]
        xe12 = self.relu(self.e12(xe11))  # -> [batch, 32, 1, time]
        xp1 = self.pool1(xe12)            # -> [batch, 32, 1, time/2]
        xp1 = self.dropout(xp1)

        xe21 = self.relu(self.e21(xp1))   # -> [batch, 64, 1, time/2]
        xe22 = self.relu(self.e22(xe21))  # -> [batch, 64, 1, time/2]
        xp2 = self.pool2(xe22)            # -> [batch, 64, 1, time/4]
        xp2 = self.dropout(xp2)

        # ------------------------
        # Bridge (Recurrent)
        # ------------------------
        # Remove spatial dim and reorder for LSTM
        xp2_squeezed = xp2.squeeze(2)                # -> [batch, 64, T]
        xp2_squeezed = xp2_squeezed.permute(0, 2, 1)  # -> [batch, T, 64]

        # Pass through LSTM
        lstm_out, _ = self.lstm(xp2_squeezed)         # -> [batch, T, lstm_hidden]

        # Project back to 64 channels
        lstm_out = lstm_out.permute(0, 2, 1)          # -> [batch, lstm_hidden, T]
        lstm_out = self.lstm_out_projection(lstm_out)  # -> [batch, 64, T]

        # Add dropout
        lstm_out = self.dropout(lstm_out)

        # Reshape back to [batch, 64, 1, T]
        xb2 = lstm_out.unsqueeze(2)

        # ------------------------
        # Decoder
        # ------------------------
        # Level 1
        xu1 = self.upconv1(xb2)    # -> [batch, 64, 1, time/2]
        if xu1.size(-1) != xe22.size(-1):
            # Pad if there's a size mismatch
            diff = xe22.size(-1) - xu1.size(-1)
            xu1 = F.pad(xu1, [0, diff, 0, 0])
        xu11 = torch.cat([xu1, xe22], dim=1)  # -> [batch, 128, 1, time/2]
        xd11 = self.relu(self.d11(xu11))      # -> [batch, 64, 1, time/2]
        xd12 = self.relu(self.d12(xd11))      # -> [batch, 64, 1, time/2]
        xd12 = self.dropout(xd12)

        # Level 2
        xu2 = self.upconv2(xd12)   # -> [batch, 32, 1, time]
        if xu2.size(-1) != xe12.size(-1):
            diff = xe12.size(-1) - xu2.size(-1)
            xu2 = F.pad(xu2, [0, diff, 0, 0])
        xu22 = torch.cat([xu2, xe12], dim=1)  # -> [batch, 64, 1, time]
        xd21 = self.relu(self.d21(xu22))      # -> [batch, 32, 1, time]
        xd22 = self.relu(self.d22(xd21))      # -> [batch, 32, 1, time]

        # ------------------------
        # Output
        # ------------------------
        out = self.outconv(xd22)          # -> [batch, 102, 1, time]
        out = self.final_upsample(out)    # -> [batch, 102, 1, 275]
        out = out.squeeze(2)              # -> [batch, 102, 275]

        return out