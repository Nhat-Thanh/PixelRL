from mini_batch_loader import *
from pixelwise_a3c import *
from MyFCN import *
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=100, help="-")
parser.add_argument("--batch_size", type=int, default=64, help="-")
parser.add_argument("--snapshot_episodes", type=int, default=100, help="-")
parser.add_argument("--test_episodes", type=int, default=100, help="-")

FLAGS, unparsed = parser.parse_known_args()


#_/_/_/ paths _/_/_/ 
TRAINING_DATA_PATH          = "../training_BSD68.txt"
TESTING_DATA_PATH           = "../testing_BSD68.txt"
IMAGE_DIR_PATH              = "../"
CKPT_PATH            = "./checkpoint"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = FLAGS.batch_size 
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = FLAGS.episodes 
EPISODE_LEN = 15
SNAPSHOT_EPISODES  = FLAGS.snapshot_episodes 
TEST_EPISODES = FLAGS.test_episodes 
GAMMA = 0.95 # discount factor

N_ACTIONS = 9
MOVE_RANGE = 3
CROP_SIZE = 70

GPU_ID = 0

def test(loader, agent):
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
    
    total_reward = sum_reward*255/test_data_size
    total_psnr = sum_psnr/test_data_size
    print("test total reward {:0.4f}, PSNR {:0.4f}".format(total_reward, total_psnr))
    return total_reward, total_psnr
    
 
def main():
    REWARD_LOG = []
    if os.path.exists(f"{CKPT_PATH}/reward_log.npy"):
        REWARD_LOG = np.load(f"{CKPT_PATH}/reward_log.npy").tolist()

    PSNR_LOG = []
    if os.path.exists(f"{CKPT_PATH}/psnr_log.npy"):
        PSNR_LOG = np.load(f"{CKPT_PATH}/psnr_log.npy").tolist()

    CUR_EPISODE = np.int32(0)
    if os.path.exists(f"{CKPT_PATH}/current_episode.npy"):
        CUR_EPISODE = np.load(f"{CKPT_PATH}/current_episode.npy")

    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH, 
        TESTING_DATA_PATH, 
        IMAGE_DIR_PATH, 
        CROP_SIZE)
 
    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
 
    # load myfcn model
    model = MyFcn(N_ACTIONS)
    if os.path.exists(f"{CKPT_PATH}/model.npz"):
        chainer.serializers.load_npz(f"{CKPT_PATH}/model.npz", model)

    #_/_/_/ setup _/_/_/
 
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)


    agent = PixelWiseA3C_InnerState(model, optimizer, int(EPISODE_LEN/3), GAMMA)
    if os.path.exists(f"{CKPT_PATH}/optimizer.npz"):
        chainer.serializers.load_npz(f"{CKPT_PATH}/optimizer.npz", agent.optimizer)
    agent.act_deterministically = True
    agent.model.to_gpu()
    
    #_/_/_/ training _/_/_/
 
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    for epi in range(1, N_EPISODES+1):
        # display current state
        episode = epi + CUR_EPISODE
        print("episode %d" % episode)

        r = indices[i:i+TRAIN_BATCH_SIZE]
        raw_x, raw_xt = mini_batch_loader.load_training_data(r)
        current_state.reset(raw_xt)
        reward = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0
        
        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agent.act_and_train(current_state.tensor, reward)
            current_state.step(action, inner_state)
            reward = np.square(raw_x - previous_image)*255 - np.square(raw_x - current_state.image)*255
            sum_reward += np.mean(reward)*np.power(GAMMA,t)

        agent.stop_episode_and_train(current_state.tensor, reward, True)
        print("train total reward {:0.4f}".format(sum_reward*255))

        if episode % TEST_EPISODES == 0:
            #_/_/_/ testing _/_/_/
            total_reward, total_psnr = test(mini_batch_loader, agent)
            REWARD_LOG.append(total_reward)
            PSNR_LOG.append(total_psnr)

        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(CKPT_PATH)
        
        if i+TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:        
            i += TRAIN_BATCH_SIZE

        if i+2*TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        optimizer.alpha = LEARNING_RATE*((1-episode/N_EPISODES)**0.9)
 
    REWARD_LOG = np.array(REWARD_LOG)
    PSNR_LOG = np.array(PSNR_LOG)
    CUR_EPISODE += epi
    np.save(f"{CKPT_PATH}/current_episode.npy", CUR_EPISODE)    
    np.save(f"{CKPT_PATH}/reward_log.npy", REWARD_LOG)
    np.save(f"{CKPT_PATH}/psnr_log.npy", PSNR_LOG)

     
 
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
