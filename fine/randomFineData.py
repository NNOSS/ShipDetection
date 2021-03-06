from PIL import Image
import loadCSV
import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import glob
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from PIL import Image


FILEPATH = '/Data/ShipDetection/'
DATA_FILEPATH = FILEPATH + 'train/'
CSV_FILENAME = DATA_FILEPATH + 'train_ship_segmentations.csv'
TRAIN_LABEL_SAVE = FILEPATH + 'fine/' + 'train_labels_fine'
TRAIN_INPUT_SAVE = FILEPATH + 'fine/' + 'train_images_fine'
TEST_LABEL_SAVE = FILEPATH + 'fine/' + 'test_labels_fine'
TEST_INPUT_SAVE = FILEPATH + 'fine/' + 'test_images_fine'
DIVIDEND = 24
WIDTH = 768
HEIGHT = 768
NEW_HEIGHT = HEIGHT / DIVIDEND
NEW_WIDTH = WIDTH / DIVIDEND
batch_size= 2000
total_images = 104000
percent_hold = .2
number_hold = total_images * percent_hold
print('Number Validate' + str(number_hold))

def append_binary_file(file_name, bytes_):
    with open(file_name,"ab") as f:
        f.write(bytes_)

files_dict_gen = loadCSV.load_csv(CSV_FILENAME,batch_size)
file_dict = next(files_dict_gen, None)
count = 0
match_num = 0
while file_dict:
    # input_group = np.zeros((len(file_dict)*DIVIDEND*DIVIDEND,NEW_WIDTH,NEW_HEIGHT,3),dtype=np.uint8)
    # label_group = np.full((len(file_dict)*DIVIDEND*DIVIDEND,NEW_WIDTH,NEW_HEIGHT),.1,dtype=np.float32)
    input_group = []
    fine_label = []

    for i, temp in enumerate(file_dict.items()):
        k,v = temp
        try:
            image = Image.open(DATA_FILEPATH + k)
            np_image = np.array(image,dtype=np.uint8)
            for j in range(DIVIDEND*DIVIDEND):
                # x_a = (j // DIVIDEND) * NEW_WIDTH
                # y_a = (j % DIVIDEND) * NEW_HEIGHT
                # label_group[i*DIVIDEND*DIVIDEND + j] = v[x_a:x_a + NEW_WIDTH, y_a:y_a + NEW_HEIGHT]
                # input_group[i*DIVIDEND*DIVIDEND + j] = np_image[x_a:x_a + NEW_WIDTH, y_a:y_a + NEW_HEIGHT]
                x_a = (j // DIVIDEND)
                y_a = (j % DIVIDEND)
                has_boat = np.amax(v[x_a * NEW_WIDTH:x_a * NEW_WIDTH + NEW_WIDTH, y_a * NEW_HEIGHT:y_a * NEW_HEIGHT + NEW_HEIGHT])
                if has_boat:
                    input_group.append(np_image[x_a * NEW_WIDTH:x_a * NEW_WIDTH + NEW_WIDTH, y_a * NEW_HEIGHT:y_a * NEW_HEIGHT + NEW_HEIGHT])
                    fine_label.append(v[x_a * NEW_WIDTH:x_a * NEW_WIDTH + NEW_WIDTH, y_a * NEW_HEIGHT:y_a * NEW_HEIGHT + NEW_HEIGHT])
                    match_num += 1
                elif match_num > 0 and np.random.random_sample() < .03:
                    input_group.append(np_image[x_a * NEW_WIDTH:x_a * NEW_WIDTH + NEW_WIDTH, y_a * NEW_HEIGHT:y_a * NEW_HEIGHT + NEW_HEIGHT])
                    fine_label.append(v[x_a * NEW_WIDTH:x_a * NEW_WIDTH + NEW_WIDTH, y_a * NEW_HEIGHT:y_a * NEW_HEIGHT + NEW_HEIGHT])
                    match_num -= 1

            count+= 1
            if not count%100:
                print('Count: ' + str(count))
                print('match_num', match_num)
        except:
            continue
    if len(input_group):
        if count < number_hold:
            print('TEST SAVE')
            input_group = np.array(input_group, dtype=np.uint8)
            fine_label = np.array(fine_label, dtype=np.uint8)
            append_binary_file(TEST_INPUT_SAVE,input_group.tobytes())
            append_binary_file(TEST_LABEL_SAVE,fine_label.tobytes())
        else:
            print('TRAIN SAVE')
            print(len(input_group))
            input_group = np.array(input_group, dtype=np.uint8)
            fine_label = np.array(fine_label, dtype=np.uint8)
            append_binary_file(TRAIN_INPUT_SAVE,input_group.tobytes())
            append_binary_file(TRAIN_LABEL_SAVE,fine_label.tobytes())
    else:
        print('EMPTY')

    file_dict = next(files_dict_gen, None)
print('DOOOOOOOOONE')
