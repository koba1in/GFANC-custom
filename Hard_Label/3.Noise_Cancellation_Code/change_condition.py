import scipy.io as sio
import numpy as np
import torchaudio
import torch
import os

def delay_secondary_path(folder, pri_path_file, sec_path_file, pri_path_name, sec_path_name, delay=0):
    pri_path = sio.loadmat(folder+pri_path_file)
    sec_path = sio.loadmat(folder+sec_path_file)
    pri_path_argmax = pri_path[pri_path_name].argmax()
    sec_path_argmax = sec_path[sec_path_name].argmax()
    zeros = np.zeros((pri_path_argmax - sec_path_argmax + delay, 1))
    sec_path = np.concat([zeros, sec_path[sec_path_name]], axis=0)
    save_file = sec_path_file[:-4] + "_" + str(delay) + sec_path_file[-4:]
    sio.savemat(folder+save_file, {sec_path_name:sec_path})

def add_noise_to_folder(folder, save_folder, snr):
    files = os.listdir(folder)
    for file in files:
        if file.lower().endswith(".wav"):
            add_noise_to_file(folder, file, save_folder, snr)
    
def add_noise_to_file(folder, file, save_folder, snr):
    wave, sample_rate = torchaudio.load(folder+file)
    wave_rms = torch.mean(torch.square(wave))
    noise_rms = wave_rms / (10 ** (snr / 10)) 
    std = torch.sqrt(noise_rms)
    noise = torch.randn_like(wave) * std
    wave = wave + noise
    wave = torch.clamp(wave, min=-1, max=1)
    torchaudio.save(save_folder+file, wave, sample_rate)

def add_noise_to_tensor(wave, snr):
    #wave: [Batch, fs*time]
    wave_rms = torch.mean(torch.square(wave))
    noise_rms = wave_rms / (10 ** (snr / 10))
    std = torch.sqrt(noise_rms)
    noise = torch.randn_like(wave) * std
    wave = wave + noise
    wave = torch.clamp(wave, min=-1, max=1)
    return wave

def add_noise_to_ndarray(wave, snr):
    wave_rms = np.mean(np.square(wave))
    noise_rms = wave_rms / (10 ** (snr / 10))
    std = np.sqrt(noise_rms)
    rng = np.random.default_rng()
    noise = rng.normal(0, std, wave.shape)
    wave = wave + noise
    wave = np.clip(wave, -1, 1)
    return wave