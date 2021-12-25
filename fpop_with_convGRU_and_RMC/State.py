import numpy as np
import sys
import cv2

class State():
    def __init__(self, size):
        self.image = np.zeros(size,dtype=np.float32)
    
    def reset(self, x):
        self.image = x
        size = self.image.shape
        prev_state = np.zeros((size[0],64,size[2],size[3]),dtype=np.float32)
        self.tensor = np.concatenate((self.image, prev_state), axis=1)

    def set(self, x):
        temp = np.copy(x)
        temp[:,0,:,:] /= 100
        temp[:,1,:,:] /= 127
        temp[:,2,:,:] /= 127
        self.tensor[:,:self.image.shape[1],:,:] = temp

    def step(self, act, inner_state):
        bgr1 = np.copy(self.image)
        bgr1 = bgr1*0.95 + 0.5*0.05
        bgr2 = np.copy(self.image)
        bgr2 = bgr2*1.05 - 0.5*0.05
        bgr_t = np.transpose(self.image, (0,2,3,1))
        temp3 = np.zeros(bgr_t.shape, bgr_t.dtype)
        temp4 = np.zeros(bgr_t.shape, bgr_t.dtype)
        b, c, h, w = self.image.shape
        for i in range(0,b):
            if np.sum(act[i]==3) > 0:
                temp = cv2.cvtColor(bgr_t[i], cv2.COLOR_BGR2HSV)
                temp[1] *= 0.95
                temp3[i] = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
            if np.sum(act[i]==4) > 0:
                temp = cv2.cvtColor(bgr_t[i], cv2.COLOR_BGR2HSV)
                temp[1] *= 1.05
                temp4[i] = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
        bgr3 = np.transpose(temp3, (0,3,1,2))
        bgr4 = np.transpose(temp4, (0,3,1,2))
        bgr5 = np.copy(self.image)
        bgr5 = bgr5 - 0.5*0.05
        bgr6 = np.copy(self.image)
        bgr6 = bgr6 + 0.5*0.05
        bgr7 = np.copy(self.image)
        bgr7[:,1:,:,:] *= 0.95
        bgr8 = np.copy(self.image)
        bgr8[:,1:,:,:] *= 1.05
        bgr9 = np.copy(self.image)
        bgr9[:,:2,:,:] *= 0.95
        bgr10 = np.copy(self.image)
        bgr10[:,:2,:,:] *= 1.05
        bgr11 = np.copy(self.image)
        bgr11[:,::2,:,:] *= 0.95
        bgr12 = np.copy(self.image)
        bgr12[:,::2,:,:] *= 1.05

        act_3channel = np.stack([act,act,act],axis=1)
        self.image = np.where(act_3channel==1, bgr1, self.image)
        self.image = np.where(act_3channel==2, bgr2, self.image)
        self.image = np.where(act_3channel==3, bgr3, self.image)
        self.image = np.where(act_3channel==4, bgr4, self.image)
        self.image = np.where(act_3channel==5, bgr5, self.image)
        self.image = np.where(act_3channel==6, bgr6, self.image)
        self.image = np.where(act_3channel==7, bgr7, self.image)
        self.image = np.where(act_3channel==8, bgr8, self.image)
        self.image = np.where(act_3channel==9, bgr9, self.image)
        self.image = np.where(act_3channel==10, bgr10, self.image)
        self.image = np.where(act_3channel==11, bgr11, self.image)
        self.image = np.where(act_3channel==12, bgr12, self.image)

        #self.image = np.maximum(self.image, 0)
        #self.image = np.minimum(self.image, 1)

        self.tensor[:,:self.image.shape[1],:,:] = self.image
        self.tensor[:,-64:,:,:] = inner_state


