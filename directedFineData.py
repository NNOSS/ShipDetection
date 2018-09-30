from PIL import Image
import loadCSV
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import glob
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
from PIL import Image


FILEPATH = '/Data/ShipDetection/'
DATA_FILEPATH = FILEPATH + 'train/'
CSV_FILENAME = FILEPATH + 'train_ship_segmentations.csv'
MODEL_FILEPATH = '/Models/CoarseShipsTrained/model.ckpt' #filepaths to model and summaries
SUMMARY_FILEPATH ='/Models/CoarseShipsTrained/Summaries/'
TEST_LABEL_SAVE = FILEPATH + 'test_labels_fine'
TEST_INPUT_SAVE = FILEPATH + 'test_images_fine'
TRAIN_LABEL_SAVE = FILEPATH + 'train_labels_fine'
TRAIN_INPUT_SAVE = FILEPATH + 'train_images_fine'
DIVIDEND = 12
WIDTH = 768
HEIGHT = 768
NEW_HEIGHT = HEIGHT / DIVIDEND
NEW_WIDTH = WIDTH / DIVIDEND
batch_size= 2000
total_images = 104000
percent_hold = .2
number_hold = total_images * percent_hold
print('Number Validate' + str(number_hold))
BASE_INPUT_SHAPE = 768, 768, 3
BASE_OUTPUT_SHAPE = 768, 768, 1
INPUT_SHAPE = BASE_INPUT_SHAPE[0]/DIVIDEND, BASE_INPUT_SHAPE[1]/DIVIDEND, 3
OUTPUT_SHAPE = BASE_OUTPUT_SHAPE[0]/DIVIDEND, BASE_OUTPUT_SHAPE[1]/DIVIDEND, 1
BATCH_SIZE = 16
CONVOLUTIONS_COARSE = [64, 128, 256, 512]
COURSE_SHAPE = DIVIDEND, DIVIDEND

def build_coarse_model(x):
    conv_pointers = [InputLayer(x, name= 'c_disc_inputs')]
    for i,v in enumerate(CONVOLUTIONS_COARSE):
        curr_layer = BatchNormLayer(Conv2d(conv_pointers[-1],
            CONVOLUTIONS_COARSE[i], (5, 5),strides = (2,2), name=
            'c_conv1_%s'%(i)),
            act=tf.nn.leaky_relu,is_train=True ,name=
            'c_batch_norm%s'%(i))

        if i < len(CONVOLUTIONS_COARSE)-1:
            # conv_pointers.append(ConcatLayer([curr_layer, conv_pointers[-1]],
            #  3, name = 'concat_layer%s'%(i)))
            conv_pointers.append(curr_layer)
        else:
            conv_pointers.append(curr_layer)
    # y_conv = DenseLayer(flat, m.fully_connected_size, act=tf.nn.relu,name =  'hidden_encode')
    pre_max_pool = Conv2d(conv_pointers[-1],
        1, (5, 5),strides = (1,1), name='c_Final_Conv')
    _, pm_width, pm_height, _ = pre_max_pool.outputs.get_shape()
    max_pool_width, max_pool_height = pm_width/DIVIDEND, pm_height/DIVIDEND
    max_pool = MaxPool2d(pre_max_pool, filter_size = (max_pool_width, max_pool_height), strides = (max_pool_width, max_pool_height), name = 'c_Final_Pool')
    logits = FlattenLayer(max_pool).outputs
    final_guess = tf.round(tf.sigmoid(max_pool.outputs))

    return final_guess

def append_binary_file(file_name, bytes_):
    with open(file_name,"ab") as f:
        f.write(bytes_)


def save_data():
    files_dict_gen = loadCSV.load_csv(CSV_FILENAME,batch_size)
    file_dict = next(files_dict_gen, None)
    sess = tf.Session()#start the session
    ##############GET DATA###############
    test_coarse_input = tf.placeholder(tf.float32, shape = (None, BASE_INPUT_SHAPE[0], BASE_INPUT_SHAPE[1], BASE_INPUT_SHAPE[2]))
    coarse_guess = build_coarse_model(test_coarse_input)
    sess.run(tf.global_variables_initializer())
    saver_perm = tf.train.Saver()
    saver_perm.restore(sess, MODEL_FILEPATH)
    count = 0
    while file_dict:
        # input_group = np.zeros((len(file_dict)*DIVIDEND*DIVIDEND,NEW_WIDTH,NEW_HEIGHT,3),dtype=np.uint8)
        # label_group = np.full((len(file_dict)*DIVIDEND*DIVIDEND,NEW_WIDTH,NEW_HEIGHT),.1,dtype=np.float32)
        input_group = []
        fine_label = []
        for k,v in file_dict.items():
            try:
                image = Image.open(DATA_FILEPATH + k)
                np_image = np.array(image,dtype=np.uint8)
                feed_dict = {test_coarse_input: np.expand_dims(np_image,0)}
                coarse_guess_ex = np.squeeze(sess.run(coarse_guess,feed_dict=feed_dict))
                for j in range(DIVIDEND*DIVIDEND):
                    x_a = (j // DIVIDEND)
                    y_a = (j % DIVIDEND)
                    # has_boat = np.amax(coarse_guess_ex[x_a * NEW_WIDTH:x_a * NEW_WIDTH + NEW_WIDTH, y_a * NEW_HEIGHT:y_a * NEW_HEIGHT + NEW_HEIGHT])

                    if coarse_guess_ex[x_a, y_a]:
                        input_group.append(np_image[x_a * NEW_WIDTH:x_a * NEW_WIDTH + NEW_WIDTH, y_a * NEW_HEIGHT:y_a * NEW_HEIGHT + NEW_HEIGHT])
                        fine_label.append(v[x_a * NEW_WIDTH:x_a * NEW_WIDTH + NEW_WIDTH, y_a * NEW_HEIGHT:y_a * NEW_HEIGHT + NEW_HEIGHT])

                count+= 1
                if not count%500:
                    print('Count: ' + str(count))
            except:
                print('Error')
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


if __name__ == "__main__":
    save_data()
