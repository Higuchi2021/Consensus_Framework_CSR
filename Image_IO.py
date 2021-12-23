import cupy as np
from PIL import Image
import Util
import matplotlib.pyplot as plt

def Image_In(K, *, is_test = False):
    if is_test:
        ims = np.zeros((K, 128*128))
        for i in range(K):
            k = i + 101
            if k < 10:
                filename = "foreman00" + str(k)
            elif k < 100:
                filename = "foreman0" + str(k)
            else:
                filename = "foreman" + str(k)
            ims[i,:] = Util.normalize(np.array(Image.open("data/foreman/"+ filename +".png"), dtype=np.float).flatten())
        return ims
    else:
        ims = np.zeros((K, 128 * 128))
        for i in range(K):
            k = i + 1
            if k < 10:
                filename = "foreman00" + str(k)
            elif k < 100:
                filename = "foreman0" + str(k)
            else:
                filename = "foreman" + str(k)
            ims[i,:] = Util.normalize(np.array(Image.open("data/foreman/" + filename + ".png"), dtype=np.float).flatten())
        return ims
        

def Examples_In(K, is_test = False):
    if is_test:
        ims = np.zeros((K, 128 * 128))
        for i in range(K):
            ims[i,:] = Util.normalize(np.array(Image.open("data/test/data0"+str(i)+".png"), dtype=np.float).flatten())
        return ims
    else :
        ims = np.zeros((K, 128 * 128))
        for i in range(K):
            ims[i,:] = Util.normalize(np.array(Image.open("data/train/data0"+str(i)+".png"), dtype=np.float).flatten())
        return ims


