import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGtoMEGUNet(nn.Module):
    """
    A simplified U-Net style model for EEG-to-MEG prediction.
    This model is intentionally smaller to reduce the likelihood of overfitting,
    but includes a recurrent layer (LSTM) in the 'bridge' to capture temporal dependencies.
    """

    def __init__(self):
        super().__init__()
        
        # ------------------------
        # Encoder
        # ------------------------
        # First Level
        self.e11 = nn.Conv2d(1, 32, kernel_size=(74, 3), padding=(0, 1))
        self.e12 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Second Level
        self.e21 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        self.e22 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # ------------------------
        # Bridge (recurrent layer)
        # ------------------------
        # Instead of a fully connected MLP bridging, we use LSTM to capture temporal dependencies
        # xp2 will be [batch, 64, 1, T], so we reshape to [batch, T, 64] for the LSTM.
        self.lstm_hidden = 32
        self.lstm = nn.LSTM(
            input_size=64,    # matches "channels" from xp2
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        # We'll project the LSTM hidden dim (32) back to 64 so the decoder has the same channels
        self.lstm_out_projection = nn.Conv1d(self.lstm_hidden, 64, kernel_size=1)

        # ------------------------
        # Decoder
        # ------------------------
        # Deconvolution layers symmetric to the encoder
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=(1, 2), stride=(1, 2))
        self.d11 = nn.Conv2d(64 + 64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.d12 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(1, 2), stride=(1, 2))
        self.d21 = nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1))
        self.d22 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1))

        # Output layer
        self.outconv = nn.Conv2d(32, 102, kernel_size=(1, 1))

        # Final upsampling to match target size
        self.final_upsample = nn.Upsample(size=(1, 275), mode='bilinear', align_corners=False)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the simplified U-Net for EEG-to-MEG.
        Expected input shape: [batch, channels=1, time=275].
        Returns: [batch, 102, time=275].
        """

        # Add a spatial dimension => [batch, 1, EEG_channels=74, time=275]
        # (We assume your EEG_channels dimension is 74, matching the kernel (74,3). If it's different, adjust accordingly.)
        x = x.unsqueeze(1)

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
        # xp2 is [batch, 64, 1, T], so let's remove the dummy spatial dim and reorder to [batch, T, 64].
        xp2_squeezed = xp2.squeeze(2)                # -> [batch, 64, T]
        xp2_squeezed = xp2_squeezed.permute(0, 2, 1)  # -> [batch, T, 64]

        # Pass through LSTM
        lstm_out, _ = self.lstm(xp2_squeezed)         # -> [batch, T, self.lstm_hidden]

        # Project back to 64 channels for decoding
        # First permute to [batch, self.lstm_hidden, T] for a 1D conv
        lstm_out = lstm_out.permute(0, 2, 1)          # -> [batch, self.lstm_hidden, T]
        lstm_out = self.lstm_out_projection(lstm_out) # -> [batch, 64, T]

        # Add dropout before decoding
        lstm_out = self.dropout(lstm_out)

        # Reshape back to [batch, 64, 1, T]
        xb2 = lstm_out.unsqueeze(2)  # -> [batch, 64, 1, T]

        # ------------------------
        # Decoder
        # ------------------------
        # Level 1
        xu1 = self.upconv1(xb2)    # -> [batch, 64, 1, time/2]
        if xu1.size(-1) != xe22.size(-1):
            # Pad if there's a mismatch in time dimension
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
        out = self.final_upsample(out)    # -> [batch, 102, 1, 275] upsampled to full time
        out = out.squeeze(2)              # -> [batch, 102, 275]

        return out