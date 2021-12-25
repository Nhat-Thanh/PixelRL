import collections
import os
import numpy as np
import cv2
import _pickle
from sklearn.datasets import fetch_20newsgroups
from PIL import ImageFont, ImageDraw, Image
 
#PathInfo = collections.namedtuple('PathInfo', ['image_path'])
 
 
class MiniBatchLoader(object):
 
    def __init__(self, train_path, test_path, image_dir_path, crop_size):
 
        # load data paths
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        self.testing_path_infos = self.read_paths(test_path, image_dir_path)
 
        self.crop_size = crop_size
        #twenty_train = fetch_20newsgroups(subset='train')
        #self.num_mask = len(twenty_train.data)
        self.num_mask = 11314

 
    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path
 
    # test ok
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c
 
    # test ok
    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs
 
    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, indices, augment=True)
 
    def load_testing_data(self, indices):
        return self.load_data(self.testing_path_infos, indices)
 
    # test ok
    def load_data(self, path_infos, indices, augment=False):
        mini_batch_size = len(indices)
        in_channels = 1

        if augment:
            xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
            xs_text = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
            
            for i, index in enumerate(indices):
                path = path_infos[index]
                
                img = cv2.imread(path,0)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                h, w = img.shape

                if np.random.rand() > 0.5:
                    img = np.fliplr(img)

                if np.random.rand() > 0.5:
                    angle = 10*np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
                    img = cv2.warpAffine(img,M,(w,h))

                rand_range_h = h-self.crop_size
                rand_range_w = w-self.crop_size
                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)
                img = img[y_offset:y_offset+self.crop_size, x_offset:x_offset+self.crop_size]
                xs[i, 0, :, :] = (img/255).astype(np.float32)

                if np.random.rand() > 0.5:
                    text_value = 255
                else:
                    text_value = 0
                img_text = Image.fromarray(img)
                mask_id = np.random.randint(0,self.num_mask)
                mask = cv2.imread('../trainmask/'+str(mask_id)+'.png', 0)
                h, w = mask.shape
                if np.random.rand() > 0.5:
                    mask = np.fliplr(mask)

                if np.random.rand() > 0.5:
                    angle = 10*np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1)
                    mask = cv2.warpAffine(mask,M,(w,h))

                rand_range_h = h-self.crop_size
                rand_range_w = w-self.crop_size
                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)
                mask = mask[y_offset:y_offset+self.crop_size, x_offset:x_offset+self.crop_size]
                mask = Image.fromarray(mask)
                img_text.paste(text_value, mask)
                img_text = np.array(img_text)
                xs_text[i, 0, :, :] = (img_text/255).astype(np.float32)

                #cv2.imshow('', img_text)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

        elif mini_batch_size == 1:
            for i, index in enumerate(indices):
                path = path_infos[index]
                
                img = cv2.imread(path,0)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))

                img_text = cv2.imread(path.replace('/test', '/test_inpaint'), 0)

            h, w = img.shape
            xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            xs[0, 0, :, :] = (img/255).astype(np.float32)

            xs_text = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            xs_text[0, 0, :, :] = (img_text/255).astype(np.float32)

            

            #cv2.imshow('', img_text)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        else:
            raise RuntimeError("mini batch size must be 1 when testing")
 
        return xs, xs_text
