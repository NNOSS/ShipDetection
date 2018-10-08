import sys
sys.path.insert(0, '/home/gtower/Desktop/ML/MODELS/ShipDetection/')
from PIL import Image
import loadCSV
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage.measure


FILEPATH = '/Data/ShipDetection/'
DATA_FILEPATH = FILEPATH + 'train/'
CSV_FILENAME = DATA_FILEPATH + 'train_ship_segmentations.csv'
TRAIN_LABEL_SAVE = FILEPATH + 'coarse/train_labels'
TRAIN_INPUT_SAVE = FILEPATH + 'coarse/train_images'
TRAIN_INPUT_SAVE_COARSE = FILEPATH + 'coarse/train_images_coarse'
TEST_LABEL_SAVE = FILEPATH + 'coarse/test_labels'
TEST_INPUT_SAVE = FILEPATH + 'coarse/test_images'
TEST_INPUT_SAVE_COARSE = FILEPATH + 'coarse/test_images_coarse'
DIVIDEND = 6
num_pool = 8
WIDTH = 768
HEIGHT = 768
NEW_HEIGHT = HEIGHT / DIVIDEND
NEW_WIDTH = WIDTH / DIVIDEND
batch_size= 2000
total_images = 104000
percent_hold = .2
number_hold = total_images * percent_hold

def append_binary_file(file_name, bytes_):
    with open(file_name,"ab") as f:
        f.write(bytes_)
def save_whole():
    files_dict_gen = loadCSV.load_csv(CSV_FILENAME,batch_size)
    file_dict = next(files_dict_gen, None)
    count = 0
    while file_dict:
        input_group = np.zeros((len(file_dict),WIDTH,HEIGHT,3),dtype=np.uint8)
        coarse_label = np.zeros((len(file_dict),DIVIDEND,DIVIDEND),dtype=np.uint8)
        fine_label = np.zeros((len(file_dict),WIDTH,HEIGHT),dtype=np.uint8)
        for i, temp in enumerate(file_dict.items()):
            k,v = temp
            try:
                image = Image.open(DATA_FILEPATH + k)
                np_image = np.array(image,dtype=np.uint8)
                input_group[i] = np_image
                fine_label[i] = v
                for j in range(DIVIDEND*DIVIDEND):
                    x_a = (j // DIVIDEND)
                    y_a = (j % DIVIDEND)
                    has_boat = np.amax(v[x_a * NEW_WIDTH:x_a * NEW_WIDTH + NEW_WIDTH, y_a * NEW_HEIGHT:y_a * NEW_HEIGHT + NEW_HEIGHT])
                    coarse_label[i, x_a, y_a] = has_boat
            except:
                print('ERR-------------------')
                continue
            count+= 1
            if not count%500:
                print('Count: ' + str(count))
        if len(input_group):
            if count < number_hold:
                print('TEST SAVE')
                input_group = np.array(input_group, dtype=np.uint8)
                fine_label = np.array(fine_label, dtype=np.uint8)
                append_binary_file(TEST_INPUT_SAVE,input_group.tobytes())
                append_binary_file(TEST_LABEL_SAVE,fine_label.tobytes())
                append_binary_file(TEST_INPUT_SAVE_COARSE,coarse_label.tobytes())
            else:
                print('TRAIN SAVE')
                print(len(input_group))
                input_group = np.array(input_group, dtype=np.uint8)
                fine_label = np.array(fine_label, dtype=np.uint8)
                append_binary_file(TRAIN_INPUT_SAVE,input_group.tobytes())
                append_binary_file(TRAIN_LABEL_SAVE,fine_label.tobytes())
                append_binary_file(TRAIN_INPUT_SAVE_COARSE,coarse_label.tobytes())
        else:
            print('EMPTY')

        file_dict = next(files_dict_gen, None)
    print('DOOOOOOOOONE')

def save_part():
    files_dict_gen = loadCSV.load_csv(CSV_FILENAME,batch_size)
    file_dict = next(files_dict_gen, None)
    count = 0
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
                    has_boat = v[x_a * NEW_WIDTH:x_a * NEW_WIDTH + NEW_WIDTH, y_a * NEW_HEIGHT:y_a * NEW_HEIGHT + NEW_HEIGHT]
                    has_boat = skimage.measure.block_reduce(has_boat, (num_pool,num_pool), np.max)
                    input_group.append(np_image[x_a * NEW_WIDTH:x_a * NEW_WIDTH + NEW_WIDTH, y_a * NEW_HEIGHT:y_a * NEW_HEIGHT + NEW_HEIGHT])
                    fine_label.append(has_boat)
                count+= 1
                if not count%100:
                    print('Count: ' + str(count))
            except:
                print('ERR------------------')
                continue
        if len(input_group):
            if count < number_hold:
                print('TEST SAVE')
                input_group = np.array(input_group, dtype=np.uint8)
                fine_label = np.array(fine_label, dtype=np.uint8)
                append_binary_file(TEST_INPUT_SAVE+ '_' + str(DIVIDEND) + '_' + str(num_pool),input_group.tobytes())
                append_binary_file(TEST_LABEL_SAVE+ '_' + str(DIVIDEND) + '_' + str(num_pool),fine_label.tobytes())
            else:
                print('TRAIN SAVE')
                print(len(input_group))
                input_group = np.array(input_group, dtype=np.uint8)
                fine_label = np.array(fine_label, dtype=np.uint8)
                append_binary_file(TRAIN_INPUT_SAVE+ '_' + str(DIVIDEND) + '_' + str(num_pool),input_group.tobytes())
                append_binary_file(TRAIN_LABEL_SAVE+ '_' + str(DIVIDEND) + '_' + str(num_pool),fine_label.tobytes())
        else:
            print('EMPTY')

        file_dict = next(files_dict_gen, None)
    print('DOOOOOOOOONE')

if __name__ == "__main__":
    save_part()
