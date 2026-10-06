import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import scipy.io as sio
import torch
import torch.nn.functional as F

from Disturbance_generation import Disturbance_reference_generation_from_Afilter, Disturbance_reference_generation_from_Afilter_with_additive_noise
from FxLMS_algorithm import FxLMS, train_fxlms_algorithm
from DFT_Filter_Decompose import Creating_Filter, Filter_Decompose
from change_condition import add_noise_to_tensor

def save_mat__(FILE_NAME_PATH, Wc):
    mdict = {'Wc_v': Wc}
    savemat(FILE_NAME_PATH, mdict)


#----------------------------------------------------
# Function loading_paths_from_MAT（）
#----------------------------------------------------
def loading_paths_from_MAT(folder, subfolder, Pri_path_file_name, Sec_path_file_name):
    Primay_path_file, Secondary_path_file = os.path.join(folder, subfolder, Pri_path_file_name), os.path.join(folder,subfolder, Sec_path_file_name)

    Pri_dfs, Secon_dfs = sio.loadmat(Primay_path_file), sio.loadmat(Secondary_path_file)
    Pri_path, Secon_path = Pri_dfs['Pz1'].squeeze(), Secon_dfs['S'].squeeze()
    return Pri_path, Secon_path
from FxLMS_algorithm import SuperFastNLMS, train_fast_lms_algorithm
from FxLMS_algorithm import SuperFastFxNLMS, train_fxlms_algorithm

def fir_filter(signal, path):
    signal = signal.view(1, 1, -1)
    path = path.flip(0).view(1, 1, -1)
    y = F.conv1d(signal, path, padding=path.shape[-1] - 1)
    return y[0, 0, :len(signal.flatten())]

def make_filter(xf, dis, num_taps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xf = torch.stack([xf[n - num_taps + 1: n + 1].flip(0) for n in range(num_taps-1, len(xf))])
    d_valid = dis[num_taps-1:]
    w = torch.linalg.lstsq(Xf, d_valid).solution
    e = dis - fir_filter(xf, w)
    return e, w.cpu().numpy()

def main():

    delay = -25
    print(delay)
    FILE_NAME_PATH = 'models/Pretrained_Sub_Control_filters_delay_' + str(delay) + '.mat'
    fs = 16000
    control_filter = Creating_Filter(Len=1024, low_cut_normal_fre=20, high_cut_normal_fre=7980, fs=fs, plot=False)

    # Configurating the pre-trained control filter parameters
    T = 30
    Len_control = 1024
    subfolder = "Secondary_path_delay_" + str(delay) + "samples"
    # subfolder = "Dongyuan"
    Pri_path, Secon_path = loading_paths_from_MAT(folder='Pz and Sz', subfolder=subfolder, Pri_path_file_name='Primary_path.mat', Sec_path_file_name='Secondary_path.mat')
    # Get the filtered-x and disturbance, train the main control filter
    # Dis, Fx = Disturbance_reference_generation_from_Afilter(fs=fs, T=T, f_vector=control_filter, Pri_path=Pri_path, Sec_path=Secon_path)
    Ref, Dis, Fx = Disturbance_reference_generation_from_Afilter(fs=fs, T=T, f_vector=control_filter, Pri_path=Pri_path, Sec_path=Secon_path,)
    Erro, Wc_main = make_filter(Fx, Dis, Len_control)
    # controller = FxLMS(Len=Len_control)
    # Erro = train_fxlms_algorithm(Model=controller, Ref=Fx, Disturbance=Dis, Stepsize=0.0001) # !!! train step size
    # controller = SuperFastFxNLMS(Len_control)
    # Erro = train_fxlms_algorithm(Model=controller, Ref=Ref, FilteredRef=Fx, Disturbance=Dis, Stepsize=0.001)
    print(10*np.log10(np.mean(np.square(np.array(Erro)))/np.mean(np.square(Dis.numpy()))))
    # Wc_main = np.squeeze(controller._get_coeff_())
    save_mat__('models/Pretrained_Main_Control_filter.mat', Wc_main)

    # Drawing the noise reduction error of the main control filter
    # plt.title('The error signal of main control filter')
    # plt.plot(Erro)
    # plt.ylabel('Amplitude')
    # plt.xlabel('Time')
    # plt.grid()
    # plt.show()

    # Devide the pretrained main control filter into 15 sub filters
    sub_filters = Filter_Decompose(Filter=Wc_main, Num_subfilters=15, fs=16000)
    save_mat__(FILE_NAME_PATH, sub_filters)
    
if __name__ == "__main__":
    main()