import numpy as np
import progressbar
import torch
from scipy import signal
from torch import optim


#------------------------------------------------------------------------------
# Class: FxLMS algorithm
#------------------------------------------------------------------------------
class FxLMS:
    
    def __init__(self, Len):
        self.Wc = torch.zeros(1, Len, requires_grad=True, dtype=torch.float) # initial coefficients of filter
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
    
    def feedforward(self, Xf): # fixed reference signal passes the control filter to output the control signal
        self.Xd = torch.roll(self.Xd, shifts=1, dims=1) # roll the tensor along the given dimension
        self.Xd[0,0] = Xf # Xf: reference signal
        yt = self.Wc @ self.Xd.t() #矩阵相乘
        return yt
    
    def LossFunction(self, y, d):
        e = d-y
        return e**2, e
    
    def _get_coeff_(self):
        return self.Wc.detach().numpy()

#------------------------------------------------------------------------------
# Function: train_fxlms_algorithm()
#------------------------------------------------------------------------------
def train_fxlms_algorithm(Model, Ref, Disturbance, Stepsize=0.00000005): 

    bar = progressbar.ProgressBar(maxval=2*Disturbance.shape[0], \
        widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
    optimizer = optim.SGD([Model.Wc], lr=Stepsize) # Stepsize is learning_rate
    
    bar.start()
    Erro_signal = []
    len_data = Disturbance.shape[0]
    for itera in range(len_data):
        # Feedfoward
        xin = Ref[itera]
        dis = Disturbance[itera]
        y = Model.feedforward(xin)
        loss,e = Model.LossFunction(y, dis)
        
        # Progress shown
        bar.update(2*itera+1)
            
        # Backward
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        Erro_signal.append(e.item())
        
        # Progress shown 
        bar.update(2*itera+2)
    bar.finish()
    return Erro_signal

#------------------------------------------------------------
# Function: Generating the testing broadband noise
#------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    filter_len = 1024
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window ='hamming',fs=fs) 
    N = filter_len + Seconds*fs
    xin = np.random.randn(N)
    y   = signal.lfilter(bandpass_filter,1,xin)
    yout= y[filter_len:]
    # Standarlize 
    yout = yout/np.sqrt(np.var(yout))
    # return a tensor of [1 x sample rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)

# class SuperFastLMS:
#     def __init__(self, Len):
#         self.Len = Len
#         self.Wc = np.zeros(Len, dtype=np.float64)  
#         self.Xd = np.zeros(Len, dtype=np.float64)  
#         self.head = 0  
    
#     def feedforward(self, Xf):

#         self.head = (self.head - 1) % self.Len
#         self.Xd[self.head] = Xf
        
#         yt = np.dot(self.Wc[:self.Len - self.head], self.Xd[self.head:]) + \
#              np.dot(self.Wc[self.Len - self.head:], self.Xd[:self.head])
#         return yt
    
#     def LossFunction(self, y, d):
#         e = d - y
#         return e**2, e
    
#     def step(self, e, stepsize):
#         self.Wc[:self.Len - self.head] += stepsize * e * self.Xd[self.head:]
#         self.Wc[self.Len - self.head:] += stepsize * e * self.Xd[:self.head]
    
#     def _get_coeff_(self):
#         return self.Wc

# class SuperFastNLMS(SuperFastLMS):
#     def __init__(self, Len):
#         super().__init__(Len)

#     def step(self, e, stepsize):
#         self.Wc[:self.Len - self.head] += stepsize / (np.finfo(float).eps + np.sum(np.square(self.Xd))) * e * self.Xd[self.head:]
#         self.Wc[self.Len - self.head:] += stepsize / (np.finfo(float).eps + np.sum(np.square(self.Xd))) * e * self.Xd[:self.head]
# #------------------------------------------------------------------------------
# # 実行関数
# #------------------------------------------------------------------------------
# def train_fast_lms_algorithm(Model, Ref, Disturbance, Stepsize=0.00000005): 

#     ref_np = np.asarray(Ref, dtype=np.float64).flatten()
#     dist_np = np.asarray(Disturbance, dtype=np.float64).flatten()
    
#     len_data = dist_np.shape[0]
#     Erro_signal = np.zeros(len_data, dtype=np.float64) 

#     for itera in range(len_data):
#         xin = ref_np[itera]
#         dis = dist_np[itera]
        
#         y = Model.feedforward(xin)
#         _, e = Model.LossFunction(y, dis)
#         Model.step(e, Stepsize)
        
#         Erro_signal[itera] = e
        
#     return Erro_signal.tolist()



# class SuperFastFxNLMS:
#     def __init__(self, Len, eps=1e-12):
#         self.Len = int(Len)
#         if self.Len <= 0:
#             raise ValueError("Len must be positive")

#         self.Wc = np.zeros(self.Len, dtype=np.float64)

#         # 出力計算用の元信号履歴
#         self.X = np.zeros(self.Len, dtype=np.float64)

#         # 係数更新用の filtered reference 履歴
#         self.Xf = np.zeros(self.Len, dtype=np.float64)

#         self.head = 0
#         self.eps = float(eps)

#     def _push(self, history, value):
#         self.head = (self.head - 1) % self.Len
#         history[self.head] = value

#     def _ordered(self, history):
#         """最新サンプルから過去に向かう順序で履歴を返す。"""
#         return np.concatenate((history[self.head:], history[:self.head]))

#     def feedforward(self, x):
#         self._push(self.X, x)
#         x_history = self._ordered(self.X)
#         return float(np.dot(self.Wc, x_history))

#     def step(self, e, stepsize, x_filtered):
#         self.Xf[self.head] = x_filtered
#         xf_history = self._ordered(self.Xf)

#         power = np.dot(xf_history, xf_history)
#         mu = stepsize / (self.eps + power)

#         self.Wc += mu * e * xf_history

#     def _get_coeff_(self):
#         return self.Wc

# def train_fxlms_algorithm(
#     Model,
#     Ref,
#     FilteredRef,
#     Disturbance,
#     Stepsize=0.5,
# ):
#     ref_np = np.asarray(Ref, dtype=np.float64).flatten()
#     filtered_np = np.asarray(FilteredRef, dtype=np.float64).flatten()
#     dist_np = np.asarray(Disturbance, dtype=np.float64).flatten()

#     if not (len(ref_np) == len(filtered_np) == len(dist_np)):
#         raise ValueError("Ref, FilteredRef, and Disturbance must have equal lengths")

#     error_signal = np.zeros(len(dist_np), dtype=np.float64)

#     for i in range(len(dist_np)):
#         y = Model.feedforward(ref_np[i])
#         e = dist_np[i] - y

#         Model.step(
#             e=e,
#             stepsize=Stepsize,
#             x_filtered=filtered_np[i],
#         )
#         error_signal[i] = e

#     return error_signal.tolist()