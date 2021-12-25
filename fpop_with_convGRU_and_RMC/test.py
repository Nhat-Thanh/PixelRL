import _pickle as pickle
from mini_batch_loader import *
from chainer import serializers
from MyFCN import *
from chainer import cuda, optimizers, Variable
import sys
import math
import time
import chainerrl
import State
import os
from pixelwise_a3c import *

#_/_/_/ paths _/_/_/ 
TRAINING_DATA_PATH          = "./uniform_set_train_id.txt"
TESTING_DATA_PATH           = "./uniform_set_test_id.txt"
IMAGE_DIR_PATH              = "../dataset_foregroundpopout/"
SAVE_PATH            = "./resultimage16bit/"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 30000
EPISODE_LEN = 10
SNAPSHOT_EPISODES  = 300
TEST_EPISODES = 300
GAMMA = 0.95 # discount factor
EPISODE_BORDER     = 15000 #decreas the learning rate at this epoch

N_ACTIONS = 13
CROP_SIZE = 70

GPU_ID = 0

def bgr2lab_tensor_converter(src):
    b, c, h, w = src.shape
    src_t = np.transpose(src, (0,2,3,1))
    dst = np.zeros(src_t.shape, src_t.dtype)
    for i in range(0,b):
        dst[i] = cv2.cvtColor(src_t[i], cv2.COLOR_BGR2Lab)
    return np.transpose(dst, (0,3,1,2))

def test(loader, agent, fout):
    sum_l2_error     = 0
    sum_reward = 0
    n_pixels = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE,3,CROP_SIZE,CROP_SIZE))
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_y, raw_x = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        current_state.reset(raw_x)
        current_image_lab = bgr2lab_tensor_converter(current_state.image)
        #reward = np.zeros(raw_x.shape, raw_x.dtype)
        
        for t in range(0, EPISODE_LEN):
            current_state.set(current_image_lab)
            action, inner_state = agent.act(current_state.tensor)
            current_state.step(action, inner_state)
            current_image_lab = bgr2lab_tensor_converter(current_state.image)

        agent.stop_episode()
            
        raw_y = np.transpose(raw_y[0], (1,2,0))
        #raw_y = np.round(raw_y*255)/255
        raw_y = cv2.cvtColor(raw_y, cv2.COLOR_BGR2Lab)
        h, w, c = raw_y.shape
        n_pixels += h*w
        current_state.image = np.transpose(current_state.image[0], (1,2,0))
        current_state.image = np.maximum(current_state.image, 0)
        current_state.image = np.minimum(current_state.image, 1)
        #current_state.image = np.round(current_state.image*255)/255
        u16image = (current_state.image*(2**16-1)+0.5).astype(np.uint16)
        cv2.imwrite(SAVE_PATH+str(i)+'.png', u16image)
        current_state.image = cv2.cvtColor(current_state.image, cv2.COLOR_BGR2Lab)
        sum_l2_error += np.sum(np.sqrt(np.sum(np.square(current_state.image-raw_y),axis=2)))/(h*w)
 
    print("test total reward {a}, l2_error {b}".format(a=sum_reward/test_data_size, b=sum_l2_error/test_data_size))
    fout.write("test total reward {a}, l2_error {b}\n".format(a=sum_reward/test_data_size, b=sum_l2_error/test_data_size))
    sys.stdout.flush()
 
 
def main(fout):
    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH, 
        TESTING_DATA_PATH, 
        IMAGE_DIR_PATH, 
        CROP_SIZE)
 
    chainer.cuda.get_device_from_id(GPU_ID).use()

    #current_state = State.State((TRAIN_BATCH_SIZE,3,CROP_SIZE,CROP_SIZE))
 
    # load myfcn model
    model = MyFcn(N_ACTIONS)
 
    #_/_/_/ setup _/_/_/
 
    #q_func = q_func.to_gpu()
    #optimizer = chainer.optimizers.RMSprop(lr=LEARNING_RATE)
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    #q_func.conv7.W.update_rule.hyperparam.alpha = 0.001
    #q_func.conv7.b.update_rule.hyperparam.alpha = 0.001

    agent = PixelWiseA3C_InnerState(model, optimizer, int(EPISODE_LEN/2), GAMMA)
    serializers.load_npz('./model/fpop_myfcn_30000/model.npz', agent.model)
    agent.act_deterministically = True
    agent.model.to_gpu()

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    #_/_/_/ testing _/_/_/
    test(mini_batch_loader, agent, fout)
 
 
     
 
if __name__ == '__main__':
    try:
        fout = open('testlog.txt', "w")
        start = time.time()
        main(fout)
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start)/60))
        print("{s}[h]".format(s=(end - start)/60/60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start)/60))
        fout.write("{s}[h]\n".format(s=(end - start)/60/60))
        fout.close()
    except Exception as error:
        print(error.message)
