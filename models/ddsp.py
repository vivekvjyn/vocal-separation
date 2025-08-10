import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(1, embedding_dim // 4, 9, padding=4)
        self.conv2 = nn.Conv1d(embedding_dim // 4, embedding_dim // 2, 19, padding=9)
        self.conv3 = nn.Conv1d(embedding_dim // 2, embedding_dim, 29, padding=14)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.mean(dim=-1)

class Harmonics(nn.Module):
    def __init__(self, embedding_dim, n_harmonics, num_samples, sr=16000):
        super(Harmonics, self).__init__()
        self.sr = sr
        self.num_samples = num_samples
        self.n_harmonics = n_harmonics
        self.fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.fc2 = nn.Linear(embedding_dim // 2, embedding_dim // 4)
        self.fc3 = nn.Linear(embedding_dim // 4, self.n_harmonics)

    def forward(self, x, f0):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        A = F.tanh(x)
        t = torch.linspace(0, self.num_samples / self.sr, self.num_samples, device=x.device)
        k = torch.arange(1, self.n_harmonics + 1, device=x.device)
        phi = (f0 * k)[:, :, None] * t[None, None, :]
        y = A[:, :, None] * torch.sin(2 * torch.pi * phi)
        y = y.sum(dim=1)
        return y

class Noise(nn.Module):
    def __init__(self, embedding_dim, num_samples):
        super(Noise, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim * 2)
        self.fc3 = nn.Linear(embedding_dim * 2, num_samples)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.tanh(x)
        return x

class Reverb(nn.Module):
    def __init__(self):
        super(Reverb, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, 9, padding=4)
        self.conv2 = nn.Conv1d(1, 1, 19, padding=9)
        self.conv3 = nn.Conv1d(1, 1, 29, padding=14)

    def forward(self, x):
        x = self.conv1(x)
        x = F.tanh(x)
        x = self.conv2(x)
        x = F.tanh(x)
        x = self.conv3(x)
        x = F.tanh(x)
        return x

class DDSP(nn.Module):
    def __init__(self, embedding_dim=256, n_harmonics=32, num_samples=1024):
        super(DDSP, self).__init__()
        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.harmonics = Harmonics(embedding_dim=embedding_dim, num_samples=num_samples, n_harmonics=n_harmonics)
        self.noise = Noise(embedding_dim=embedding_dim, num_samples=num_samples)
        self.reverb = Reverb()

    def forward(self, x, f0):
        z = self.encoder(x)
        sine = self.harmonics(z, f0)
        eps = self.noise(z)
        y = sine + eps
        x = self.reverb(y.unsqueeze(1))
        return x
