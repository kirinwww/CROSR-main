import torch
import torch.nn as nn
import torch.nn.functional as F


class DHRNet1D(nn.Module):

    def __init__(self, num_classes, input_channels=1, base_channels=64, latent_channels=32, hidden_dim=512):
        super(DHRNet1D, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

        self.conv1_1 = nn.Conv1d(input_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm1d(base_channels)
        self.conv1_2 = nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm1d(base_channels)

        self.conv2_1 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm1d(base_channels * 2)
        self.conv2_2 = nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm1d(base_channels * 2)

        self.conv3_1 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm1d(base_channels * 4)
        self.conv3_2 = nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm1d(base_channels * 4)
        self.conv3_3 = nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1)
        self.bn3_3 = nn.BatchNorm1d(base_channels * 4)

        self.btl1 = nn.Conv1d(base_channels, latent_channels, kernel_size=3, stride=1, padding=1)
        self.btlu1 = nn.Conv1d(latent_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.btl2 = nn.Conv1d(base_channels * 2, latent_channels, kernel_size=3, stride=1, padding=1)
        self.btlu2 = nn.Conv1d(latent_channels, base_channels * 2, kernel_size=3, stride=1, padding=1)
        self.btl3 = nn.Conv1d(base_channels * 4, latent_channels, kernel_size=3, stride=1, padding=1)
        self.btlu3 = nn.Conv1d(latent_channels, base_channels * 4, kernel_size=3, stride=1, padding=1)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc4 = nn.Linear(base_channels * 4, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, self.num_classes)

        self.deconv3 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=2, stride=2, padding=0)
        self.deconv1 = nn.ConvTranspose1d(base_channels, input_channels, kernel_size=2, stride=2, padding=0)

    def _resize_like(self, x, reference):
        if x.size(-1) != reference.size(-1):
            x = F.interpolate(x, size=reference.size(-1), mode='linear', align_corners=False)
        return x

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("DHRNet1D expects input with shape [batch, channels, length]")

        input_length = x.size(-1)

        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)))
        x1 = F.max_pool1d(x1, kernel_size=2, stride=2)
        x1 = F.dropout(x1, p=0.25, training=self.training)

        x2 = F.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))
        x2 = F.max_pool1d(x2, kernel_size=2, stride=2)
        x2 = F.dropout(x2, p=0.25, training=self.training)

        x3 = F.relu(self.bn3_1(self.conv3_1(x2)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))
        x3 = F.relu(self.bn3_3(self.conv3_3(x3)))
        x3 = F.max_pool1d(x3, kernel_size=2, stride=2)
        x3 = F.dropout(x3, p=0.25, training=self.training)

        x4 = self.global_pool(x3).flatten(start_dim=1)

        x5 = F.dropout(F.relu(self.fc4(x4)), p=0.5, training=self.training)
        x5 = F.dropout(F.relu(self.fc5(x5)), p=0.5, training=self.training)
        x5 = self.fc6(x5)

        z3 = F.relu(self.btl3(x3))
        z2 = F.relu(self.btl2(x2))
        z1 = F.relu(self.btl1(x1))

        j3 = self.btlu3(z3)
        j2 = self.btlu2(z2)
        j1 = self.btlu1(z1)

        g2 = F.relu(self.deconv3(j3))
        g2 = self._resize_like(g2, x2)

        g1 = F.relu(self.deconv2(j2 + g2))
        g1 = self._resize_like(g1, x1)

        g0 = self.deconv1(j1 + g1)
        g0 = self._resize_like(g0, x)
        g0 = g0[..., :input_length]

        return x5, g0, [z3, z2, z1]
