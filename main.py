#%%
import cupy as np
import Options
import Util
import Image_IO
from Coef_update_L2_fast import Coef_update_L2
from Dict_update_L2 import Dict_update_L2
import Visualize
import Decode_Random
import Encode
from tqdm import tqdm 

#%% Preparation of dictionary learning
np.random.seed(seed=123)
dictlearn_opt = Options.DictLearn_Option(K=20, M=10, N=128*128, L=32*32, Image_Width=128, Filter_Width=32, Lambda=0.9, Rho=100.0, coef_iteration=2, dict_iteration=2)#0.6:80
D = np.random.normal(0, 1, (dictlearn_opt.K, dictlearn_opt.M, dictlearn_opt.Filter_Width, dictlearn_opt.Filter_Width))
#D = Util.dict_vectorize(D, dictlearn_opt)
D = Util.dict_vectorize_2Dlike(D, dictlearn_opt)
X = np.zeros((dictlearn_opt.K, dictlearn_opt.M, dictlearn_opt.N))
S = Image_IO.Image_In(dictlearn_opt.K)
#S = S.reshape((dictlearn_opt.K, 128, 128))[:,50:78,50:78].reshape((dictlearn_opt.K, dictlearn_opt.N))
S = S.reshape((dictlearn_opt.K, 128, 128)).reshape((dictlearn_opt.K, dictlearn_opt.N))

Visualize.get_tiled_Img(S, dictlearn_opt)

#print("DXSshape")
#print(D.shape)
#print(X.shape)
#print(S.shape)
#print(S[0,1])

Visualize.get_tiled_Xk(X[0], dictlearn_opt)
#Visualize.get_tiled_D(D, dictlearn_opt)
Visualize.get_tiled_D_2Dlike(D[0], dictlearn_opt)

#%% Dictionary learning
Coef = Coef_update_L2(dictlearn_opt, S)
Dict = Dict_update_L2(dictlearn_opt, S)

for i in tqdm(range(1500)):
#for i in range(100):
    #print("{}th coef updating is below".format(i+1))
    X = Coef.coef_update_L2(D)
    #print()
    #print("{}th dict updating is below".format(i+1))
    D = Dict.dict_update_L2(X)
    #print()
    if(i % 500 == 0):
    #    Coef.get_tiled_X()
    #    Dict.get_tiled_D()
    #    Dict.testget_tiled_D()
        Visualize.get_tiled_Xk(X[0], dictlearn_opt)
        #Visualize.get_tiled_D(D, dictlearn_opt)
        Visualize.get_tiled_D_2Dlike(D[0], dictlearn_opt)

#Visualize.get_tiled_Xk(X[0], dictlearn_opt)
#Visualize.get_tiled_D(D, dictlearn_opt)
#Dict.testget_tiled_D()


#%%
#for i in range(dictlearn_opt.K):
#    Visualize.get_tiled_Xk(X[i], dictlearn_opt, titlename=' {}th Result Coefficient'.format(i))
Visualize.get_tiled_D_2Dlike(D[0], dictlearn_opt, titlename='Result Dictionary')    
#Visualize.get_tiled_D(D, dictlearn_opt)

#%%
print("against {} pixels".format(dictlearn_opt.N))
Visualize.get_L0_norm(X, dictlearn_opt)
print(X[0,0,0])
print(D[2,0,0])

#/////////////////////////////////////////////////////////
#作成した辞書でのリコンストラクト
#/////////////////////////////////////////////////////////
#%%
dictlearn_opt.coef_iteration = 5000
#dictlearn_opt.Lambda = 2
S_test = Image_IO.Image_In(dictlearn_opt.K)
#S_test = S_test.reshape((dictlearn_opt.K, 128, 128))[:,50:78,50:78].reshape((dictlearn_opt.K, dictlearn_opt.N))
S_test = S_test.reshape((dictlearn_opt.K, 128, 128)).reshape((dictlearn_opt.K, dictlearn_opt.N))
Coef = Coef_update_L2(dictlearn_opt, S)
X_ = np.zeros((dictlearn_opt.K, dictlearn_opt.M, dictlearn_opt.N))
X_ = Coef.coef_update_L2(D)
#for i in range(dictlearn_opt.K):
Visualize.get_tiled_Xk(X_[0], dictlearn_opt)
Visualize.get_tiled_D_2Dlike(D[0], dictlearn_opt)
S_dx = Visualize.get_reconstructed_img(D,X_,dictlearn_opt)

#%%
print("against {} pixels".format(dictlearn_opt.N))
Visualize.get_L0_norm(X_, dictlearn_opt)
Visualize.get_psnr_and_ssim(S, S_dx, dictlearn_opt)

#/////////////////////////////////////////////////////////
#テストデータのカスカス表現
#/////////////////////////////////////////////////////////
#%%
dictlearn_opt.coef_iteration = 5000
#dictlearn_opt.Lambda = 0.1
S_test = Image_IO.Examples_In(10, True)
#S_test = S_test.reshape((dictlearn_opt.K, 128, 128))[:,50:78,50:78].reshape((dictlearn_opt.K, dictlearn_opt.N))
S_test = S_test.reshape((dictlearn_opt.K, 128, 128)).reshape((dictlearn_opt.K, dictlearn_opt.N))
Coef = Coef_update_L2(dictlearn_opt, S_test)
X_ = np.zeros((dictlearn_opt.K, dictlearn_opt.M, dictlearn_opt.N))
X_ = Coef.coef_update_L2(D)
#for i in range(dictlearn_opt.K):
Visualize.get_tiled_Xk(X_[0], dictlearn_opt, titlename='Coefficients_test')
S_dx = Visualize.get_reconstructed_img(D,X_,dictlearn_opt, titlename='Reconstructed_test')


#%%
print("against {} pixels".format(dictlearn_opt.N))
Visualize.get_L0_norm(X_, dictlearn_opt)
Visualize.get_psnr_and_ssim(S_test, S_dx, dictlearn_opt)



#/////////////////////////////////////////////////////////
#ランダム行列を用いた観測行列での復号(L2)
#/////////////////////////////////////////////////////////
# %%
#for i in range(40):
rrhhoo = 10
decode_random_opt = Options.Decode_Random_Option(R=128*128 // 2, Myu=0, Rho=rrhhoo, iteration=200, dictlearn_opt=dictlearn_opt)
NKF = Image_IO.Image_In(decode_random_opt.K, is_test=True)
NKF = NKF.reshape(-1,)


decode_random = Decode_Random.Decode_Random(decode_random_opt, NKF)

Y = Encode.encode_random(NKF, decode_random_opt)

decode_X = decode_random.decode(D[0], Y)
decode_img =  Util.convolutional_sum(D[0], decode_X, decode_random_opt)
print("Myu: {}, Rho: {}, Myu/Rho: {}, against {} pixels".format(decode_random_opt.Myu, decode_random_opt.Rho, decode_random_opt.Myu / decode_random_opt.Rho, decode_random_opt.N))
Visualize.get_L0_norm(decode_X.reshape(1,*decode_X.shape), decode_random_opt)
Visualize.get_tiled_Img(decode_img.reshape(1,-1), decode_random_opt, titlename='decoded image')
Visualize.get_psnr_and_ssim(NKF.reshape(1,-1), decode_img.reshape(1,-1), decode_random_opt)



#%%

print(128*128)

#%%












print(X_[0,0,0])
print(28*28*5)
print(28*28*6)
#学習時の係数のL0ノルム
Visualize.get_L0_norm(X, dictlearn_opt)





#%%














#%%
target = np.array([1,2,1,3])
image = np.array([[[0,0],[1,1]],[[2,2],[3,3]],[[4,4],[5,5]],[[6,6],[7,7]]])
print(target.shape)
print(image.shape)
print("image")
print(image)
print("a")
print(image[np.where(target == 1)])

#%%
a = np.array([1,2,3,4])
b = np.tile(a, (2,1))
print(b)

c = np.array([a,a])
print(c)
print(c.shape)
d = np.tile(c, (3,1,1))

print(d)
print(d.shape)

# %%
a = np.ones((5,6,10))
b = np.sum(a, 1)
print(b.shape)
print(b)

c = np.zeros((1,6,10))
d = c * a
print(d.shape)
print(d)

print(np.ones((4,), dtype=complex))