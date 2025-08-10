from tracemalloc import Frame
import librosa
from models import ddsp
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from scipy.signal import get_window
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    mix, sr = librosa.load('mix.wav', sr=44100, mono=True)
    mix = mix[:10 * 44100]
    target, _ = librosa.load('target.wav', sr=44100, mono=True)
    target = target[:10 * 44100]

    f0, _, _ = librosa.pyin(target, sr=sr, hop_length=256, frame_length=1024, fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C7'))
    f0[np.isnan(f0)] = 0.0

    mix = librosa.util.frame(mix, frame_length=1024, hop_length=256).T
    target = librosa.util.frame(target, frame_length=1024, hop_length=256).T
    f0 = torch.tensor(f0[:len(mix)], dtype=torch.float32).unsqueeze(1).to(device)


    mix = torch.tensor(mix, dtype=torch.float32).unsqueeze(1).to(device)
    target = torch.tensor(target, dtype=torch.float32).unsqueeze(1).to(device)
    train_dataset = torch.utils.data.TensorDataset(mix, target, f0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = ddsp.DDSP()

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(100):
        total_loss = 0.0
        print(f'Epoch [{epoch+1}/100]:')
        model.train()
        for i, (mix, target, f0) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(mix, f0)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'\tLoss: {total_loss / len(train_loader):.8f}')

    output = model(mix, f0)
    audio = np.zeros(output.shape[0] * output.shape[2])
    window = get_window('hann', 1024)
    for i, row in enumerate(output):
        audio[i * 256 : i * 256 + 1024] = row.detach().cpu().numpy().squeeze() * window
    sf.write('output.wav', audio, sr)



if __name__ == "__main__":
    main()
