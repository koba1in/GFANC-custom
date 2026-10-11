# Fixed filter noise cancellation by Sample

import torch
import torch.nn.functional as F

class Fixed_filter_controller:
    def __init__(self, Filter_vector, fs):
        self.Filter_vector = torch.from_numpy(Filter_vector).type(torch.float)# torch.Size([Xseconds, 1024])
        Len = self.Filter_vector.shape[1]
        self.fs = fs
        self.Xd = torch.zeros(1, Len, dtype=torch.float)
        self.Current_Filter = torch.zeros(1, Len, dtype=torch.float)
    
    def noise_cancellation(self, Dis, Fx):
        Erro = torch.zeros(Dis.shape[0])
        j = 0
        for ii, dis in enumerate(Dis):
            self.Xd = torch.roll(self.Xd,1,1)
            self.Xd[0,0] = Fx[ii] # Fx[ii]: fixed-x signal
            yt = self.Current_Filter @ self.Xd.t()
            e = dis - yt
            Erro[ii] = e.item()
            if (ii + 1) % self.fs == 0:
                self.Current_Filter = self.Filter_vector[j]
                j += 1
        return Erro


class My_Fixed_filter_controller(Fixed_filter_controller):
    def __init__(self, Filter_vector, fs, Secondary_path):
        super().__init__(Filter_vector, fs)
        self.Sec_path = Secondary_path

    def filter(self, signal, w):
        signal = signal.view(1, 1, -1)
        w = w.flip(0).view(1, 1, -1)
        y = F.conv1d(signal, w, padding=w.shape[-1]-1)
        return y[0, 0, :len(signal.flatten())]
    
    def noise_cancellation(self, Dis, Ref):
            # Erro = torch.zeros(Dis.shape[0])
            yts = []
            j = 0
            for ii, dis in enumerate(Dis):
                self.Xd = torch.roll(self.Xd,1,1)
                self.Xd[0,0] = Ref[ii] # Fx[ii]: fixed-x signal
                yt = self.Current_Filter @ self.Xd.t()
                yts.append(yt.item())
                # e = dis - yt
                # Erro[ii] = e.item()
                if (ii + 1) % self.fs == 0:
                    self.Current_Filter = self.Filter_vector[j]
                    j += 1
            yts = torch.tensor(yts)
            Erro = Dis - self.filter(yts, self.Sec_path)
            return Erro