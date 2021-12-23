import cupy as np
import Options
import Convert
import Util
from tqdm import tqdm


class Decode_Random():
    def __init__(self, opt, NKF):
        self.opt = opt
        self.NKF = NKF
    
    def decode(self, D, Y):
        #print(D.shape)
        #print(Y.shape)
        self.Df = Convert.FFT_MN(D, self.opt)
        self.X = np.zeros((self.opt.M, self.opt.N))
        self.Phi = self.opt.Phi
        self.Y = Y

        for i in tqdm(range(self.opt.iteration)):
            self.X_old = self.X.copy()
            self.X_update()

        return self.X

    def X_update(self):
        phiDx = self.Phi @ Util.convolutional_sum(self.Df, self.X_old, self.opt, is_D_real=False)
        nblf = Convert.IFFT_MN(np.conj(self.Df) * Convert.FFT_N(np.dot(self.Phi.T, (phiDx - self.Y)), self.opt), self.opt)
        z = self.X_old - self.opt.Rho * nblf
        self.X = self.prox_L1(z, self.opt.Myu / self.opt.Rho)

    def prox_L1(self, x, alpha):
        #print("prox_L1")
        #print(x[0,0,0])
        res = np.sign(x) * (np.clip(np.abs(x) - alpha, 0, float('Inf')))
        #print(res[0,0,0])
        return res