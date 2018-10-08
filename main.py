import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import glob
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
from PIL import Image
import islandProblem


FILEPATH = '/Data/ShipDetection/'
DATA_FILEPATH = FILEPATH + 'test/'
MODEL_FILEPATH = '/Models/Ships/FinalShipsDeep/model.ckpt' #filepaths to model and summaries
SUMMARY_FILEPATH ='/Models/Ships/FinalShipsDeep/Summaries/'
SAVE_PICS = '/home/gtower/Pictures/'

CONVOLUTIONS_FINE =  [32, -32, 64, -64, 128, -128]
DECONVOLUTIONS_FINE = [-128, 128, -64, 64, -32, 32]
DIVIDEND = 24
BASE_INPUT_SHAPE = 768, 768, 3
BASE_OUTPUT_SHAPE = 768, 768, 1
INPUT_SHAPE = BASE_INPUT_SHAPE[0]/DIVIDEND, BASE_INPUT_SHAPE[1]/DIVIDEND, 3
OUTPUT_SHAPE = BASE_OUTPUT_SHAPE[0]/DIVIDEND, BASE_OUTPUT_SHAPE[1]/DIVIDEND, 1
BATCH_SIZE = 16

CONVOLUTIONS_COARSE = [-64, 128, -128, 256, -256, 512]
COURSE_SHAPE = DIVIDEND, DIVIDEND

def build_coarse_model(x):
    conv_pointers = [InputLayer(x, name= 'c_disc_inputs')]
    for i,v in enumerate(CONVOLUTIONS_COARSE):
        if v < 0:
            strides = (2,2)
            v *= -1
        else:
            strides = (1,1)
        curr_layer = BatchNormLayer(Conv2d(conv_pointers[-1],
            v, (5, 5),strides = strides, name=
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
        1, (1, 1),strides = (1,1), name='c_Final_Conv')
    _, pm_width, pm_height, _ = pre_max_pool.outputs.get_shape()
    max_pool_width, max_pool_height = pm_width/DIVIDEND, pm_height/DIVIDEND
    max_pool = MaxPool2d(pre_max_pool, filter_size = (max_pool_width, max_pool_height), strides = (max_pool_width, max_pool_height), name = 'c_Final_Pool')
    logits = FlattenLayer(max_pool).outputs
    final_guess = tf.round(tf.sigmoid(max_pool.outputs))

    return final_guess

def build_fine_model(x):
    with tf.variable_scope(tf.get_variable_scope()):
        conv_pointers = [InputLayer(x, name= 'f_disc_inputs')]
        conv_pointers_concat = []
        for i,v in enumerate(CONVOLUTIONS_FINE):
            if v < 0:
                v *= -1
                curr_layer = Conv2d(BatchNormLayer(conv_pointers[-1],
                    act=tf.nn.relu,is_train=True ,name=
                    'f_batch_norm%s'%(i)),
                    v, (5, 5),strides = (1,1), name=
                    'f_conv1_%s'%(i))
                conv_pointers_concat.append(curr_layer)
                curr_layer = MaxPool2d(curr_layer, filter_size = (2,2), strides = (2,2), name = 'pool_%s'%(i))
                conv_pointers.append(curr_layer)
            else:
                curr_layer = Conv2d(BatchNormLayer(conv_pointers[-1],
                act=tf.nn.relu,is_train=True ,name=
                'f_batch_norm%s'%(i)),
                    v, (5, 5),strides = (1,1), name=
                    'f_conv1_%s'%(i))
                conv_pointers.append(curr_layer)

        deconv_pointers = [conv_pointers[-1]]
        for i,v in enumerate(DECONVOLUTIONS_FINE):
            if v > 0:
                prev_layer = BatchNormLayer(conv_pointers_concat.pop(),
                    act=tf.nn.relu,is_train=True ,name=
                    'f_deconv_batch_t_norm%s'%(i))
                curr_layer = BatchNormLayer(deconv_pointers[-1],
                    act=tf.nn.relu,is_train=True ,name=
                    'f_deconv_batch_norm%s'%(i))
                concat_layer = InputLayer(tf.concat([prev_layer.outputs, curr_layer.outputs],3))
                deconv_pointers.append(Conv2d(concat_layer,
                    v, filter_size=(5, 5),strides = (1,1), name=
                    'f_deconv_%s'%(i)))
            else:
                v *= -1
                concat_layer = BatchNormLayer(deconv_pointers[-1],
                    act=tf.nn.relu,is_train=True ,name=
                    'f_deconv_batch_norm%s'%(i))
                deconv_pointers.append(DeConv2d(concat_layer,
                    v, filter_size=(5, 5),strides = (2,2), name=
                    'f_deconv_%s'%(i)))
        final_layer = Conv2d(deconv_pointers[-1],
            1, filter_size=(5, 5),strides = (1,1), name=
            'f_deconv_%s'%(i))
        # logits = FlattenLayer(final_layer).outputs
        final_guess = tf.round(tf.sigmoid(final_layer.outputs))
        return final_guess

if __name__ == "__main__":
    NUM_SKIP = 500
    sess = tf.Session()#start the session
    ##############GET DATA###############
    test_coarse_input = tf.placeholder(tf.float32, shape = (None, BASE_INPUT_SHAPE[0], BASE_INPUT_SHAPE[1], BASE_INPUT_SHAPE[2]))
    coarse_guess = build_coarse_model(test_coarse_input)
    test_fine_input = tf.placeholder(tf.float32, shape = (None, None, None, 3))
    fine_guess = build_fine_model(test_fine_input)

    sess.run(tf.global_variables_initializer())
    #
    saver_perm = tf.train.Saver()
    saver_perm.restore(sess, MODEL_FILEPATH)

    for name in glob.glob(DATA_FILEPATH + '*.jpg'):
        if NUM_SKIP:
            NUM_SKIP -= 1
            continue
        print('Go')
        image = Image.open(name)
        np_image = np.array(image,dtype=np.float32) / 255.0
        feed_dict = {test_coarse_input: np.expand_dims(np_image,0)}
        coarse_guess_ex = sess.run(coarse_guess,feed_dict=feed_dict)
        indices = []
        box_list = islandProblem.get_boxes(np.squeeze(coarse_guess_ex))

        # print(coarse_guess_ex)
        # if np.max(coarse_guess_ex):
        #     im = Image.fromarray(np.squeeze(coarse_guess_ex)* 255)
        #     im = im.resize((BASE_INPUT_SHAPE[0],BASE_INPUT_SHAPE[1]))
        #     im.show()

        # for j in range(DIVIDEND*DIVIDEND):
        #     # x_a = (j // DIVIDEND) * NEW_WIDTH
        #     # y_a = (j % DIVIDEND) * NEW_HEIGHT
        #     # label_group[i*DIVIDEND*DIVIDEND + j] = v[x_a:x_a + NEW_WIDTH, y_a:y_a + NEW_HEIGHT]
        #     # input_group[i*DIVIDEND*DIVIDEND + j] = np_image[x_a:x_a + NEW_WIDTH, y_a:y_a + NEW_HEIGHT]
        #     x_a = (j // DIVIDEND)
        #     y_a = (j % DIVIDEND)
        #     # print(coarse_guess_ex)
        #     has_boat = np.squeeze(coarse_guess_ex)[x_a,y_a]
        #     if has_boat:
        #         run_fine_model.append(np_image[x_a * INPUT_SHAPE[0]:x_a * INPUT_SHAPE[0] + INPUT_SHAPE[0], y_a * INPUT_SHAPE[1]:y_a * INPUT_SHAPE[1] + INPUT_SHAPE[1]])
        #         indices.append(j)
        if len(box_list):
            full_guess = np.zeros((BASE_OUTPUT_SHAPE[0],BASE_OUTPUT_SHAPE[1]))
            print(box_list)
            for mins,maxes in box_list:
                x_min, y_min = mins[0], mins[1]
                x_max, y_max = maxes[0]+1, maxes[1]+1
                fine_ex = np_image[x_min * INPUT_SHAPE[0]:x_max * INPUT_SHAPE[0], y_min * INPUT_SHAPE[1]:y_max * INPUT_SHAPE[1]]
                fine_guesses = np.squeeze(sess.run(fine_guess, feed_dict= {test_fine_input: np.expand_dims(fine_ex, 0)}))
                full_guess[x_min * INPUT_SHAPE[0]:x_max * INPUT_SHAPE[0], y_min * INPUT_SHAPE[1]:y_max * INPUT_SHAPE[1]] = fine_guesses
            full_guess = np.tile(np.expand_dims(full_guess, -1), [1,1,3])
            con = np.uint8(np.concatenate([np_image* 255, full_guess* 255], axis = 1))
            print(con.shape)
            print(type(con))
            im = Image.fromarray(con, 'RGB')
            im.show()
            raw_input()




    # train_model(head_block , ITERATIONS, test_bool)
