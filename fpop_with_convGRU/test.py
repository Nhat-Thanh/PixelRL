from mini_batch_loader import *
from pixelwise_a3c import *
from MyFCN import *
import chainer
import State
import time
import os
# import _pickle as pickle
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
TRAINING_DATA_PATH          = "../uniform_set_train_id.txt"
TESTING_DATA_PATH           = "../uniform_set_test_id.txt"
IMAGE_DIR_PATH              = "../dataset_foregroundpopout/"
CKPT_PATH = FLAGS.ckpt_path
SAVE_PATH            = "./resultimage16bit"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TEST_BATCH_SIZE  = 1 #must be 1
EPISODE_LEN = 10
GAMMA = 0.95 # discount factor

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

def test(loader, agent):
    sum_l2_error     = 0
    sum_reward = 0
    n_pixels = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE,3,CROP_SIZE,CROP_SIZE))
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_y, raw_x = loader.load_testing_data(np.array(range(i, i+TEST_BATCH_SIZE)))
        current_state.reset(raw_x)
        current_image_lab = bgr2lab_tensor_converter(current_state.image)
        
        for t in range(0, EPISODE_LEN):
            current_state.set(current_image_lab)
            action, inner_state = agent.act(current_state.tensor)
            current_state.step(action, inner_state)
            current_image_lab = bgr2lab_tensor_converter(current_state.image)

        agent.stop_episode()
            
        raw_y = np.transpose(raw_y[0], (1,2,0))
        raw_y = cv2.cvtColor(raw_y, cv2.COLOR_BGR2Lab)
        h, w, c = raw_y.shape
        n_pixels += h*w
        current_state.image = np.transpose(current_state.image[0], (1,2,0))
        current_state.image = np.maximum(current_state.image, 0)
        current_state.image = np.minimum(current_state.image, 1)
        u16image = (current_state.image*(2**16-1)+0.5).astype(np.uint16)
        
        cv2.imwrite(f"{SAVE_PATH}/{i}.png", u16image)
        
        current_state.image = cv2.cvtColor(current_state.image, cv2.COLOR_BGR2Lab)
        sum_l2_error += np.sum(np.sqrt(np.sum(np.square(current_state.image-raw_y),axis=2)))/(h*w)
    
    total_reward = sum_reward/test_data_size
    l2_error = sum_l2_error/test_data_size
    print("test total reward {:0.4f}, l2_error {:0.4f}".format(total_reward, l2_error))
    return total_reward, l2_error

 
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
 
    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState(model, optimizer, int(EPISODE_LEN/2), GAMMA)
    if os.path.exists(f"{CKPT_PATH}/optimizer.npz"):
        chainer.serializers.load_npz(f"{CKPT_PATH}/optimizer.npz", agent.optimizer)
    agent.act_deterministically = True
    agent.model.to_gpu()

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

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
