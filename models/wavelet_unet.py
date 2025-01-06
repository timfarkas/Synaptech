import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGtoMEGUNet(nn.Module):
    """
    A U-Net style model for EEG-to-MEG prediction using wavelet-transformed data.
    Input: [batch, 3, time] (3 wavelet bands from EEG)
    Output: [batch, 3, time] (3 wavelet bands for MAG prediction)
    """

    def __init__(self):
        super().__init__()

        # ------------
        # Encoder
        # ------------
        self.e11 = nn.Conv2d(3, 32, kernel_size=(1, 3), padding=(0, 1))  # Changed in_channels to 3
        self.e12 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        self.e21 = nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1))
        self.e22 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # ------------
        # Bridge (LSTM)
        # ------------
        self.lstm_hidden = 32
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.lstm_out_projection = nn.Conv1d(self.lstm_hidden, 64, kernel_size=1)

        # ------------
        # Decoder
        # ------------
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=(1, 2), stride=(1, 2))
        self.d11 = nn.Conv2d(64 + 64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.d12 = nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(1, 2), stride=(1, 2))
        self.d21 = nn.Conv2d(32 + 32, 32, kernel_size=(1, 3), padding=(0, 1))
        self.d22 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding=(0, 1))

        # Changed out_channels to 3 for wavelet bands
        self.outconv = nn.Conv2d(32, 3, kernel_size=(1, 1))

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the U-Net.
        Args:
            x: Input tensor of shape [batch, 3, time]
        Returns:
            Output tensor of shape [batch, 3, time]
        """
        # Add spatial dimension for 2D convolutions
        x = x.unsqueeze(2)  # [batch, 3, 1, time]

        # Encoder Path
        xe11 = self.relu(self.e11(x))
        xe12 = self.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)
        xp1 = self.dropout(xp1)

        xe21 = self.relu(self.e21(xp1))
        xe22 = self.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)
        xp2 = self.dropout(xp2)

        # Bridge (LSTM)
        xp2_squeezed = xp2.squeeze(2)
        xp2_squeezed = xp2_squeezed.permute(0, 2, 1)
        lstm_out, _ = self.lstm(xp2_squeezed)
        lstm_out = lstm_out.permute(0, 2, 1)
        lstm_out = self.lstm_out_projection(lstm_out)
        lstm_out = self.dropout(lstm_out)
        xb2 = lstm_out.unsqueeze(2)

        # Decoder Path
        xu1 = self.upconv1(xb2)
        if xu1.shape[-1] != xe22.shape[-1]:
            diff = xe22.shape[-1] - xu1.shape[-1]
            xu1 = F.pad(xu1, [0, diff, 0, 0])
        xu1_cat = torch.cat([xu1, xe22], dim=1)
        xd11 = self.relu(self.d11(xu1_cat))
        xd12 = self.relu(self.d12(xd11))
        xd12 = self.dropout(xd12)

        xu2 = self.upconv2(xd12)
        if xu2.shape[-1] != xe12.shape[-1]:
            diff = xe12.shape[-1] - xu2.shape[-1]
            xu2 = F.pad(xu2, [0, diff, 0, 0])
        xu2_cat = torch.cat([xu2, xe12], dim=1)
        xd21 = self.relu(self.d21(xu2_cat))
        xd22 = self.relu(self.d22(xd21))

        # Output
        out = self.outconv(xd22)  # [batch, 3, 1, time]
        out = out.squeeze(2)      # [batch, 3, time]
        return out