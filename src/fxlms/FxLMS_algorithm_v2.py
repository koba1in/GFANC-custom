import numpy as np
import progressbar
import torch
from scipy import signal
from torch import optim
import torch.nn.functional as F
from common.utils import fir_filter

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

class Noisy_FxLMS(FxLMS):
    
    def __init__(self, Len):
        super().__init__(Len)
        self.Wc = torch.zeros(1, Len, requires_grad=False, dtype=torch.float)

    def update(self, e, fx, Stepsize):
        # Fx: shape [1, Len] のFiltered-xベクトル
        self.Wc += 2 * Stepsize * e.detach() * fx.flip(0).unsqueeze(0)

def train_fxlms_algorithm_for_noisy_ref(Model, Ref, Disturbance, Secondary_path, filter_len, Stepsize=0.00000005): 
    #Ref: reference signal
    bar = progressbar.ProgressBar(maxval=2*Disturbance.shape[0], \
        widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        
    bar.start()
    Erro_signal = []
    len_data = Disturbance.shape[0]
    Fx = fir_filter(Ref, Secondary_path)
    Fx = torch.concat([torch.zeros(filter_len - 1), Fx])
    Y = torch.zeros(len(Secondary_path))
    Secondary_path = Secondary_path #.flip(0)
    for itera in range(len_data):
        # Feedfoward
        xin = Ref[itera]
        dis = Disturbance[itera]
        y = Model.feedforward(xin)
        Y = torch.roll(Y, shifts=1, dims=0)
        Y[0] = y
        y_secondary = torch.dot(Y, Secondary_path)
        loss, e = Model.LossFunction(y_secondary, dis)
        
        # Progress shown
        bar.update(2*itera+1)
        if not torch.isfinite(e).all():
            print("non-finite e at", itera)
        if not torch.isfinite(Fx).all():
            print("non-finite fx at", itera)

        # Backward 
        Model.update(e, Fx[itera:itera+filter_len], Stepsize)
        Erro_signal.append(e.item())
        if not torch.isfinite(Model.Wc).all():
            print("non-finite Wc after update at", itera)
            break
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


# class NumpyFxLMS:
    
#     def __init__(self, Len):
        
#         self.Wc = np.zeros((1, Len), dtype=np.float64) # initial coefficients of filter
#         self.Xd = np.zeros((1, Len), dtype=np.float64)
    
#     def feedforward(self, Xf): # fixed reference signal passes the control filter to output the control signal
#         self.Xd = np.roll(self.Xd, shift=1, axis=1) # roll the tensor along the given dimension
#         self.Xd[0,0] = Xf # Xf: reference signal
#         yt = self.Wc @ self.Xd.T
#         return yt
    
#     def LossFunction(self, y, d):
#         e = d-y
#         return e**2, e
    
#     def step(self, e, stepsize):
#         self.Wc += stepsize * e.item() * self.Xd
        
#     def _get_coeff_(self):
#         return self.Wc

# #------------------------------------------------------------------------------
# # Function: train_fxlms_algorithm()
# #------------------------------------------------------------------------------
# def train_numpy_fxlms_algorithm(NumpyModel, Ref, Disturbance, Stepsize=0.00000005): 

#     # bar = progressbar.ProgressBar(maxval=2*Disturbance.shape[0], \
#     #     widgets = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    
#     # bar.start()
#     Erro_signal = []
#     len_data = Disturbance.shape[0]
#     for itera in range(len_data):
#         # Feedfoward
#         xin = Ref[itera]
#         dis = Disturbance[itera]
#         y = NumpyModel.feedforward(xin)
#         loss,e = NumpyModel.LossFunction(y, dis)
#         NumpyModel.step(e, Stepsize)
#         # Progress shown
#         # bar.update(2*itera+1)
            
#         Erro_signal.append(e.item())
        
        # Progress shown 
        # bar.update(2*itera+2)
    # bar.finish()
#     return Erro_signal
# class SuperFastLMS():
#     def __init__(self, Len):
#         self.Len = Len
#         self.Wc = np.zeros(Len, dtype=np.float32)  
#         self.Xd = np.zeros(Len, dtype=np.float32)  
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



# class SuperFastFxLMS:
#     def __init__(self, Len, sec_path, sec_path_hat=None, epsilon=1e-8):
#         if Len <= 0:
#             raise ValueError("Len must be positive")

#         self.Len = Len
#         self.Wc = np.zeros(Len, dtype=np.float64)
#         self.Xd = np.zeros(Len, dtype=np.float64)   # 基準信号の履歴
#         self.Xf = np.zeros(Len, dtype=np.float64)   # filtered-x の履歴
#         self.head = 0
#         self.epsilon = epsilon

#         self.sec_path = np.asarray(sec_path, dtype=np.float64).flatten()
#         self.sec_path_hat = np.asarray(
#             sec_path if sec_path_hat is None else sec_path_hat,
#             dtype=np.float64,
#         ).flatten()

#         # 実際の二次経路と、その推定モデルのフィルタ状態
#         self._sec_zi = np.zeros(max(len(self.sec_path) - 1, 0))
#         self._hat_zi = np.zeros(max(len(self.sec_path_hat) - 1, 0))

#         self._last_y = 0.0
#         self._last_xf_power = 0.0

#     def _push(self, buffer, value):
#         self.head = (self.head - 1) % self.Len
#         buffer[self.head] = value

#     def _dot_history(self, coeffs, history):
#         split = self.Len - self.head
#         return (
#             np.dot(coeffs[:split], history[self.head:])
#             + np.dot(coeffs[split:], history[:self.head])
#         )

#     def feedforward(self, x):
#         """基準入力を受け、二次経路通過後の制御音を返す。"""
#         self._push(self.Xd, x)

#         # 制御フィルタの出力。適応フィルタには未フィルタの x を入力。
#         self._last_y = self._dot_history(self.Wc, self.Xd)

#         # filtered-x を作り、係数更新用履歴に保存する。
#         xf_arr, self._hat_zi = signal.lfilter(
#             self.sec_path_hat, [1.0], [x], zi=self._hat_zi
#         )
#         xf = xf_arr[0]
#         self._push(self.Xf, xf)
#         self._last_xf_power = float(np.dot(self.Xf, self.Xf))

#         # 制御出力が実際の二次経路を通った信号を返す。
#         y_sec_arr, self._sec_zi = signal.lfilter(
#             self.sec_path, [1.0], [self._last_y], zi=self._sec_zi
#         )
#         return y_sec_arr[0]

#     def LossFunction(self, y, d):
#         e = d - y
#         return e**2, e

#     def step(self, e, stepsize):
#         # NLMS 更新。係数には filtered-x の履歴を使う。
#         mu = stepsize / (self.epsilon + self._last_xf_power)
#         split = self.Len - self.head
#         self.Wc[:split] += mu * e * self.Xf[self.head:]
#         self.Wc[split:] += mu * e * self.Xf[:self.head]


# def train_fast_lms_algorithm(Model, Ref, Disturbance, Stepsize=0.00000005):
#     ref_np = np.asarray(Ref, dtype=np.float64).flatten()
#     dist_np = np.asarray(Disturbance, dtype=np.float64).flatten()

#     if ref_np.size != dist_np.size:
#         raise ValueError("Ref and Disturbance must have the same length")

#     error_signal = np.zeros(ref_np.size, dtype=np.float64)

#     for i, (x, d) in enumerate(zip(ref_np, dist_np)):
#         y_sec = Model.feedforward(x)
#         _, e = Model.LossFunction(y_sec, d)
#         Model.step(e, Stepsize)
#         error_signal[i] = e

#     return error_signal.tolist()