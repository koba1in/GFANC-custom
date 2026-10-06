import torch
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import scipy.signal as signal
import scipy.io as sio


#----------------------------------------------------------------
# Function: SFANC_FxNLMS
# Description: Using FxNLMS to optimize the control filter, the initial weights come from SFANC
#----------------------------------------------------------------
class SFANC_FxNLMS():
    def __init__(self, MAT_FILE, fs):
        self.Wc = self.Load_Pretrained_filters_to_tensor(MAT_FILE) # torch.Size([15, 1024])
        Len = self.Wc.shape[1]
        self.fs = fs
        self.Current_Filter = torch.zeros(1, Len, dtype=torch.float)
    
    def noise_cancellation(self, Dis, Fx, filter_index):
        Error = []
        j = 0
        
        for ii, dis in enumerate(Dis):
            y,power = model.feedforward(Fx[ii])
            loss,e = model.LossFunction(y,dis,power)
            
            Error.append(e.item())
            
            if (ii + 1) % self.fs == 0:
                print(j)
                if self.Current_Filter[0].equal(self.Wc[filter_index[j]]) == False: 
                    # if prediction index is changed, change initial weights of FxNLMS
                    print('change the initial weights of FxNLMS')
                    self.Current_Filter = self.Wc[filter_index[j]].unsqueeze(0) # torch.Size([1, 1024])
                j += 1
        return Error
        
    def Load_Pretrained_filters_to_tensor(self, MAT_FILE): # Loading the pre-trained control filter from the mat file
        mat_contents = sio.loadmat(MAT_FILE)
        Wc_vectors = mat_contents['Wc_v']
        return torch.from_numpy(Wc_vectors).type(torch.float)