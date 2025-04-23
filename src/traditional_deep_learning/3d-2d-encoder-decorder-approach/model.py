import torch
import torch.nn as nn
from tqdm import trange


class Encoder3D(nn.Module):
    """3D Encoder that extracts features from (batch, channels, t, h, w) to a fixed size."""

    def __init__(self, in_channels=1, base_channels=32, latent_dim=256):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Adaptive pooling to ensure a fixed output shape (independent of t, h, w)
        self.global_pool = nn.AdaptiveAvgPool3d(
            (1, 4, 4)
        )  # Reduce t to 1, h and w to 4x4

        # Fully connected layer to get a fixed feature vector
        self.fc = nn.Linear(base_channels * 4 * 4 * 4, latent_dim)

    def forward(self, x):
        """
        Input: (batch, latent_dim, t, h, w) where t is variable
        Output: (batch, out_features) - Fixed-size feature vector
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.global_pool(x)  # Shape: (batch, C, 1, 4, 4)
        x = x.view(x.shape[0], -1)  # Flatten to (batch, C*4*4)

        x = self.fc(x)  # Final feature vector
        return x


class Decoder2D(nn.Module):
    """Decoder that takes fixed-size features and reconstructs (batch, channels, h, w)."""

    def __init__(
        self, latent_dim=256, base_channels=32, out_channels=1, output_size=(128, 128)
    ):
        super().__init__()

        self.output_size = output_size  # Target (h, w)

        # FC layer to reshape back to a spatial representation
        self.fc = nn.Linear(
            latent_dim, base_channels * 4 * 16 * 16
        )  # Expands to (batch, C, 4, 4)

        # Transpose Conv layers to progressively upscale
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(inplace=True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),  # Assuming the output is normalized (0-1 range)
        )

    def forward(self, x):
        """
        Input: (batch, latent_dim) - Encoded feature vector
        Output: (batch, channels, h, w) - Reconstructed image
        """
        x = self.fc(x)  # Shape: (batch, C*8*4*4)
        x = x.view(x.shape[0], -1, 16, 16)  # Reshape to (batch, C, 4, 4)
        # print(x.shape)

        x = self.deconv1(x)  # (batch, C/2, 8, 8)
        # print(x.shape)
        x = self.deconv2(x)  # (batch, C/4, 16, 16)
        # print(x.shape)
        x = self.deconv3(x)  # (batch, out_channels, 128, 128)

        return x


class EncoderDecoder(nn.Module):
    def __init__(
        self, in_channels=1, base_channels=32, latent_dim=256, output_size=(128, 128)
    ):
        super().__init__()

        # Encoder: Extracts fixed-size features from (batch, channels, t, h, w)
        self.encoder = Encoder3D(in_channels, base_channels, latent_dim)

        # Decoder: Predicts (batch, channels, h, w) from features
        self.decoder = Decoder2D(
            latent_dim,
            base_channels * 2,
            out_channels=in_channels,
            output_size=output_size,
        )

    def forward(self, x):
        """
        Training forward pass:
        - Iterates over `num_slices` time steps.
        - Uses `input[:, :, t:t+i, :, :]` to predict `input[:, :, t+1, :, :]`
        - Returns stacked predicted slices.

        Input shape: (batch, channels, T, H, W)
        Output shape: (batch, channels, num_slices, H, W)
        """
        T = x.shape[2]
        outputs = []

        for t in range(1, T):
            # Select a time window (batch, channels, i, h, w)
            input_slice = x[:, :, :t, :, :]

            # Encode the time-slice into features
            features = self.encoder(input_slice)

            # Decode back to an image slice
            predicted_slice = self.decoder(features)  # (batch, channels, h, w)

            outputs.append(predicted_slice.unsqueeze(2))  # Add time dimension

        return torch.cat(outputs, dim=2)  # (batch, channels, num_slices, h, w)

    def predict(self, x, n_slices):
        """
        Autoregressive prediction:
        - Uses `x_init[:, :, 0, :, :]` as the first frame.
        - Iteratively predicts the next frame, appends it to input, and repeats.

        Input:
            x_init: (batch, channels, 1, H, W)  # First frame
            num_predictions: Number of future slices to generate

        Output: (batch, channels, num_predictions, H, W)
        """
        predicted_slices = [x]  # Store initial input

        for _ in trange(1, n_slices + 1):
            # Stack previous predictions into (batch, channels, T, h, w)
            input_sequence = torch.stack(
                predicted_slices, dim=2
            )  # (batch, channels, T, h, w)

            # Encode & decode
            features = self.encoder(input_sequence)
            next_prediction = self.decoder(features)  # (batch, channels, h, w)

            # Append next frame
            predicted_slices.append(next_prediction)

        return torch.stack(
            predicted_slices[1:], dim=2
        )  # (batch, channels, num_predictions, H, W)


if __name__ == "__main__":
    # Example Usage
    batch_size, channels, T, H, W = 2, 1, 10, 128, 128

    model = EncoderDecoder(in_channels=channels, base_channels=32, latent_dim=256)
    input_tensor = torch.randn(batch_size, channels, T, H, W)

    # Training forward pass
    output = model(input_tensor)  # (batch, channels, num_slices, h, w)
    print("Training Output Shape:", output.shape)  # Expect (2, 1, num_slices, 128, 128)

    # Autoregressive prediction
    x_init = input_tensor[:, :, 0:1, :, :]  # (batch, channels, 1, h, w)
    preds = model.predict(x_init, n_slices=5)
    print("Autoregressive Output Shape:", preds.shape)  # Expect (2, 1, 5, 128, 128)
