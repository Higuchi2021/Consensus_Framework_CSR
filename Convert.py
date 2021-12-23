import cupy as np
import Options

#///////////////////////////////////////////////////////////
#FFT to each vector of MN matrix
#///////////////////////////////////////////////////////////
def FFT_N(D, opt):
    Df = np.fft.fft2(D.reshape(opt.Image_Width, opt.Image_Width)).reshape(-1,)
    return Df 
def IFFT_N(Df, opt):
    D = np.fft.ifft2(Df.reshape(opt.Image_Width, opt.Image_Width)).reshape(-1,)
    return D
def FFT_MN(D, opt):
    Df = np.fft.fft2(D.reshape(opt.M, opt.Image_Width, opt.Image_Width)).reshape(opt.M, -1)
    return Df 
def IFFT_MN(Df, opt):
    D = np.fft.ifft2(Df.reshape(opt.M, opt.Image_Width, opt.Image_Width)).reshape(opt.M, -1)
    return D
def FFT_KN(D, opt):
    Df = np.fft.fft2(D.reshape(opt.K, opt.Image_Width, opt.Image_Width)).reshape(opt.K, -1)
    return Df 
def IFFT_KN(Df, opt):
    D = np.fft.ifft2(Df.reshape(opt.K, opt.Image_Width, opt.Image_Width)).reshape(opt.K, -1)
    return D
def FFT_KMN(D, opt):
    Df = np.fft.fft2(D.reshape(opt.K, opt.M, opt.Image_Width, opt.Image_Width)).reshape(opt.K, opt.M, -1)
    return Df 
def IFFT_KMN(Df, opt):
    D = np.fft.ifft2(Df.reshape(opt.K, opt.M, opt.Image_Width, opt.Image_Width)).reshape(opt.K, opt.M, -1)
    return D


