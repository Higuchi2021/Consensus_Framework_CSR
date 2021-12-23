import cupy as np
import Options
import Convert
import Util
import sys
from scipy.linalg import lu_factor, lu_solve

class Dict_update_L2():
    def __init__(self, opt, S):
        self.opt = opt
        self.S = S
        self.Sf = Convert.FFT_KN(self.S, self.opt)

        self.X = np.zeros((self.opt.K, self.opt.M, self.opt.N))
        self.G = np.zeros((self.opt.M, self.opt.N))
        self.H = np.zeros((self.opt.K, self.opt.M, self.opt.N))

    def dict_update_L2(self, X):

        self.X = X
        self.Xf = Convert.FFT_KMN(self.X, self.opt)
        #print(self.D.shape)
        #print(self.Df.shape)
        
        for i in range(self.opt.dict_iteration):
            self.H_old = self.H.copy()

            self.D_update()
            self.G_update()
            self.H_update()

        return np.tile(self.G, (self.opt.K, 1, 1))

    def D_update(self):
        XfH = np.conj(self.Xf)
        Gf = Convert.FFT_MN(self.G, self.opt)
        Hf = Convert.FFT_KMN(self.H, self.opt)

        #print("D update processing...")
        a = XfH / (self.opt.Rho * np.ones((self.opt.N), dtype=complex) + np.sum(self.Xf*XfH, axis=1)).reshape(self.opt.K,1,self.opt.N)
        b = np.sum(self.Xf*(Gf-Hf), axis=1) - self.Sf
        c = Gf - Hf
        #broadcastのためのreshape
        b = b.reshape(self.opt.K,1,self.opt.N)

        Df = c - (a * b)

        self.D = Convert.IFFT_KMN(Df, self.opt).real

    def G_update(self):
        #self.G = self.proxICPN(self.D + self.H)
        self.G = self.proxICPN_2Dlike((1.0/self.opt.K) * np.sum(self.D + self.H_old, axis=0))
    
    def H_update(self):
        self.H = self.H_old + self.D - self.G
    
    def proxICPN(self, d):
        PPT = np.zeros((self.opt.M, self.opt.N))
        PPT[:, 0:self.opt.L] = 1
        PPTd = PPT * d
        for i in range(self.opt.M):
            if np.linalg.norm(PPTd[i,:], ord=2) > 1.0:
                PPTd[i,:] /= np.linalg.norm(PPTd[i,:], ord=2)
        return PPTd

    def proxICPN_2Dlike(self, d):
        PPT = np.zeros((self.opt.M, self.opt.Image_Width, self.opt.Image_Width))
        PPT[:,:self.opt.Filter_Width, :self.opt.Filter_Width] = np.ones((self.opt.M, self.opt.Filter_Width, self.opt.Filter_Width))
        PPT = PPT.reshape((self.opt.M, self.opt.N))
        PPTd = PPT * d
        for i in range(self.opt.M):
            if np.linalg.norm(PPTd[i,:], ord=2) > 1.0:
                PPTd[i,:] /= np.linalg.norm(PPTd[i,:], ord=2)
        return PPTd


 