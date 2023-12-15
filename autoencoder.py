import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_shape=(15, 99, 99)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0], out_channels=15, kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(
                in_channels=15, out_channels=12, kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(
                in_channels=12, out_channels=9, kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(
                in_channels=9, out_channels=6, kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Flatten(),
        )
        self.encoder = self.encoder.float()

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (6, 3, 3)),
            nn.ConvTranspose2d(in_channels=6, out_channels=9, kernel_size=3),
            nn.Upsample(size=(10, 10)),
            nn.ConvTranspose2d(in_channels=9, out_channels=12, kernel_size=3),
            nn.Upsample(size=(24, 24)),
            nn.ConvTranspose2d(in_channels=12, out_channels=15, kernel_size=3),
            nn.Upsample(size=(52, 52)),
            nn.ConvTranspose2d(in_channels=15, out_channels=15, kernel_size=3),
            nn.Upsample(size=(99, 99)),
        )
        self.decoder = self.decoder.float()

    def forward(self, features):
        emb = self.encoder(features)
        reconstructed = self.decoder(emb)
        if reconstructed.shape != features.shape:
            print(reconstructed.shape)
            print(features.shape)
            assert False
        return reconstructed