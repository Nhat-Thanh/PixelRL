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
TRAINING_DATA_PATH          = "/mnt/hdd/furuta/chainer_inpaint_share/training.txt"
TESTING_DATA_PATH           = "/mnt/hdd/furuta/chainer_inpaint_share/testing.txt"
IMAGE_DIR_PATH              = "/mnt/hdd/furuta/chainer_inpaint_share"
SAVE_PATH            = "./resultimage/"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 30000
EPISODE_LEN = 15
SNAPSHOT_EPISODES  = 300
TEST_EPISODES = 300
GAMMA = 0.95 # discount factor
EPISODE_BORDER     = 15000 #decreas the learning rate at this epoch

N_ACTIONS = 9
MOVE_RANGE = 3
CROP_SIZE = 70

GPU_ID = 0

def test(loader, agent, fout):
    sum_psnr     = 0
    sum_reward = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_x, raw_xt = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        current_state.reset(raw_xt)
        reward = np.zeros(raw_x.shape, raw_x.dtype)*255
        
        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agent.act(current_state.tensor)
            current_state.step(action, inner_state)
            reward = np.square(raw_x - previous_image)*255 - np.square(raw_x - current_state.image)*255
            sum_reward += np.mean(reward)*np.power(GAMMA,t)

        agent.stop_episode()
            
        I = np.maximum(0,raw_x)
        I = np.minimum(1,I)
        p = np.maximum(0,current_state.image)
        p = np.minimum(1,p)
        I = (I*255+0.5).astype(np.uint8)
        p = (p*255+0.5).astype(np.uint8)
        sum_psnr += cv2.PSNR(p, I)
        p = np.transpose(p[0], [1,2,0])
        cv2.imwrite(SAVE_PATH+str(i)+'.png', p)
 
    print("test total reward {a}, PSNR {b}".format(a=sum_reward*255/test_data_size, b=sum_psnr/test_data_size))
    fout.write("test total reward {a}, PSNR {b}\n".format(a=sum_reward*255/test_data_size, b=sum_psnr/test_data_size))
    sys.stdout.flush()
 
 
def main(fout):
    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH, 
        TESTING_DATA_PATH, 
        IMAGE_DIR_PATH, 
        CROP_SIZE)
 
    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
    #ra = State.RandomActor(current_state)
 
    # load myfcn model
    model = MyFcn(N_ACTIONS)
 
    #_/_/_/ setup _/_/_/
 
    #q_func = q_func.to_gpu()
    #optimizer = chainer.optimizers.RMSprop(lr=LEARNING_RATE)
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    #q_func.conv7.W.update_rule.hyperparam.alpha = 0.001
    #q_func.conv7.b.update_rule.hyperparam.alpha = 0.001

    agent = PixelWiseA3C_InnerState(model, optimizer, EPISODE_LEN, GAMMA)
    serializers.load_npz('./model/inpaint_myfcn_30000/model.npz', agent.model)
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
