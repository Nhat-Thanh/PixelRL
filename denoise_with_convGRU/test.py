from mini_batch_loader import *
from pixelwise_a3c import *
from MyFCN import *
import chainer
import State
import time
import os
# from chainer import serializers
# from chainer import cuda, optimizers, Variable
# import sys
# import math
# import chainerrl
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="./checkpoint", help="-")
FLAGS, unparsed = parser.parse_known_args()

#_/_/_/ paths _/_/_/ 
TRAINING_DATA_PATH          = "../training_BSD68.txt"
TESTING_DATA_PATH           = "../testing_BSD68.txt"
IMAGE_DIR_PATH              = "../"
CKPT_PATH = FLAGS.ckpt_path 
SAVE_PATH            = "./resultimage"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TEST_BATCH_SIZE  = 1 #must be 1
EPISODE_LEN = 5
GAMMA = 0.95 # discount factor

#noise setting
MEAN = 0
SIGMA = 15

N_ACTIONS = 9
MOVE_RANGE = 3 #number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 70

GPU_ID = 0

def test(loader, agent):
    sum_psnr     = 0
    sum_reward = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_x = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        raw_n = np.random.normal(MEAN,SIGMA,raw_x.shape).astype(raw_x.dtype)/255
        current_state.reset(raw_x,raw_n)
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
        N = np.maximum(0,raw_x+raw_n)
        N = np.minimum(1,N)
        p = np.maximum(0,current_state.image)
        p = np.minimum(1,p)
        I = (I[0]*255+0.5).astype(np.uint8)
        N = (N[0]*255+0.5).astype(np.uint8)
        p = (p[0]*255+0.5).astype(np.uint8)
        p = np.transpose(p,(1,2,0))
        I = np.transpose(I,(1,2,0))
        N = np.transpose(N,(1,2,0))
        
        cv2.imwrite(f"{SAVE_PATH}/{i}_output.png",p)
        cv2.imwrite(f"{SAVE_PATH}/{i}_input.png",N)

        sum_psnr += cv2.PSNR(p, I)

    total_reward = sum_reward*255/test_data_size
    total_psnr = sum_psnr/test_data_size
    print("test total reward {:0.4f}, PSNR {:0.4f}".format(total_reward, total_psnr))
    return total_reward, total_psnr

 
def main():
    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH, 
        TESTING_DATA_PATH, 
        IMAGE_DIR_PATH, 
        CROP_SIZE)
 
    chainer.cuda.get_device_from_id(GPU_ID).use()

    # load myfcn model
    model = MyFcn(N_ACTIONS)
    if os.path.exists(f"{CKPT_PATH}/model.npz"):
        chainer.serializers.load_npz(f"{CKPT_PATH}/model.npz", model)
    else:
        chainer.serializers.load_npz('./model/pretrained_15.npz', model)
 
    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState(model, optimizer, EPISODE_LEN, GAMMA)
    if os.path.exists(f"{CKPT_PATH}/optimizer.npz"):
        chainer.serializers.load_npz(f"{CKPT_PATH}/optimizer.npz", agent.optimizer)
    agent.act_deterministically = True
    agent.model.to_gpu()

    #_/_/_/ testing _/_/_/
    test(mini_batch_loader, agent)
    
     
 
if __name__ == '__main__':
    try:
        start = time.time()
        main()
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start)/60))
        print("{s}[h]".format(s=(end - start)/60/60))
        
    except Exception as error:
        print(error.message)
