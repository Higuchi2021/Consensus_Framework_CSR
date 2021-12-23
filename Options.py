import cupy as np
class DictLearn_Option():
    def __init__(self, K, M, N, L, Image_Width, Filter_Width, Lambda, Rho, coef_iteration, dict_iteration):
        """
        K:　画像の枚数
        M:  辞書の枚数
        N:  画像の画素数
        L:　辞書の画素数
        Image_Width:　画像の一辺の画素数
        Filter_Width:　辞書の一辺の画素数
        Lambda: 非ゼロ係数
        Rho:    ADMM内での係数
        coef_iteration: 係数更新の繰り返し回数
        dict_iteration: 辞書更新の繰り返し回数
        """
        self.K = K
        self.M = M
        self.N = N
        self.L = L
        self.Image_Width = Image_Width 
        self.Filter_Width = Filter_Width
        self.Lambda = Lambda
        self.Rho = Rho
        self.coef_iteration = coef_iteration
        self.dict_iteration = dict_iteration

        
class Decode_Random_Option():
    def __init__(self, R, Myu, Rho, iteration, dictlearn_opt):
        """
        R:　圧縮行列の各列ベクトルの次元
        Myu:　非ゼロ係数
        Rho:　ステップ幅
        iteration:　係数の更新回数
        """
        self.R = R
        self.Myu = Myu
        self.Rho = Rho
        self.iteration = iteration

        self.K = 1
        self.M = dictlearn_opt.M
        self.N = dictlearn_opt.N
        self.L = dictlearn_opt.L
        self.Image_Width = dictlearn_opt.Image_Width 
        self.Filter_Width = dictlearn_opt.Filter_Width

        if(R == self.N):
            self.Phi = np.eye(self.L)
        else:
            self.Phi = np.random.normal(0, 1/self.N, (self.R, self.N))
        #self.Phi = np.random.normal(0, 1, (self.L, self.N))
        