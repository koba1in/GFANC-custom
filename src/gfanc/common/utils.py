import torch.nn.functional as F

def fir_filter(signal, path):
    signal = signal.view(1, 1, -1)
    path = path.flip(0).view(1, 1, -1)
    y = F.conv1d(signal, path, padding=path.shape[-1] - 1)
    return y[0, 0, :len(signal.flatten())]