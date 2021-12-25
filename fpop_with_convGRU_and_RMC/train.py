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
SAVE_PATH            = "./model/fpop_myfcn_"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 100 
EPISODE_LEN = 10
SNAPSHOT_EPISODES  = 100 
TEST_EPISODES = 100 
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
        raw_y = np.round(raw_y*255)/255
        raw_y = cv2.cvtColor(raw_y, cv2.COLOR_BGR2Lab)
        h, w, c = raw_y.shape
        n_pixels += h*w
        current_state.image = np.transpose(current_state.image[0], (1,2,0))
        current_state.image = np.maximum(current_state.image, 0)
        current_state.image = np.minimum(current_state.image, 1)
        current_state.image = np.round(current_state.image*255)/255
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

    current_state = State.State((TRAIN_BATCH_SIZE,3,CROP_SIZE,CROP_SIZE))
 
    # load myfcn model
    model = MyFcn(N_ACTIONS)
    if os.path.exists("./model/fpop_myfcn_100/model.npz"):
        serializers.load_npz('./model/fpop_myfcn_100/model.npz', model)
    #_/_/_/ setup _/_/_/
 
    #q_func = q_func.to_gpu()
    #optimizer = chainer.optimizers.RMSprop(lr=LEARNING_RATE)
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    #q_func.conv7.W.update_rule.hyperparam.alpha = 0.001
    #q_func.conv7.b.update_rule.hyperparam.alpha = 0.001

    agent = PixelWiseA3C_InnerState(model, optimizer, int(EPISODE_LEN/2), GAMMA)
    if os.path.exists("./model/fpop_myfcn_100/optimizer.npz"):
        serializers.load_npz('./model/fpop_myfcn_100/optimizer.npz', agent.optimizer)
    agent.act_deterministically = True
    agent.model.to_gpu()
    
    #_/_/_/ training _/_/_/
 
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    for episode in range(1, N_EPISODES+1):
        print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()
        r = indices[i:i+TRAIN_BATCH_SIZE]
        raw_y, raw_x = mini_batch_loader.load_training_data(r)
        current_state.reset(raw_x)
        current_image_lab = bgr2lab_tensor_converter(current_state.image)
        raw_y_lab = bgr2lab_tensor_converter(raw_y)
        reward = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0
        
        for t in range(0, EPISODE_LEN):
            previous_image_lab = current_image_lab.copy()
            current_state.set(current_image_lab)
            action, inner_state = agent.act_and_train(current_state.tensor, reward)
            current_state.step(action, inner_state)
            current_image_lab = bgr2lab_tensor_converter(current_state.image)
            reward = np.sqrt(np.sum(np.square(raw_y_lab - previous_image_lab),axis=1)[:,np.newaxis,:,:]) - np.sqrt(np.sum(np.square(raw_y_lab - current_image_lab),axis=1)[:,np.newaxis,:,:])
            sum_reward += np.mean(reward)*np.power(GAMMA,t)

        agent.stop_episode_and_train(current_state.tensor, reward, True)
        print("train total reward {a}".format(a=sum_reward))
        fout.write("train total reward {a}\n".format(a=sum_reward))
        sys.stdout.flush()

        if episode % TEST_EPISODES == 0:
            #_/_/_/ testing _/_/_/
            test(mini_batch_loader, agent, fout)

        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(SAVE_PATH+str(episode))
        
        if i+TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:        
            i += TRAIN_BATCH_SIZE

        if i+2*TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        #if episode % EPISODE_BORDER == 0:
        #    optimizer.alpha *= 0.1
        optimizer.alpha = LEARNING_RATE*((1-episode/N_EPISODES)**0.9)
 
     
 
if __name__ == '__main__':
    try:
        fout = open('log.txt', "w")
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
