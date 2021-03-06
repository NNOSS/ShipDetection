import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

FILEPATH = '/Data/ShipDetection/'
DATA_FILEPATH = FILEPATH + 'train/'
CSV_FILENAME = DATA_FILEPATH + 'train_ship_segmentations.csv'
TRAIN_INPUT_SAVE = FILEPATH + 'coarse/train_images_6_8'
TRAIN_LABEL_SAVE_COARSE = FILEPATH + 'coarse/train_labels_6_8'
TEST_INPUT_SAVE = FILEPATH + 'coarse/test_images_6_8'
TEST_LABEL_SAVE_COARSE = FILEPATH + 'coarse/test_labels_6_8'

PERM_MODEL_FILEPATH = '/Models/Ships/CoarseShipsTrained/model.ckpt' #filepaths to model and summaries
SUMMARY_FILEPATH ='/Models/Ships/CoarseShipsTrained/Summaries/'

RESTORE = True
WHEN_DISP = 50
WHEN_SAVE = 2000
MAX_OUTPUTS = 50
ITERATIONS = 1000000
LEARNING_RATE = 3e-3
WHEN_TEST = 50


CONVOLUTIONS = [-64, 128, -128, 256, -256, 512]
DIVIDEND = 6
NUM_POOL = 8
BASE_INPUT_SHAPE = 768, 768, 3

INPUT_SHAPE = BASE_INPUT_SHAPE[0]/DIVIDEND, BASE_INPUT_SHAPE[1]/DIVIDEND, 3
OUTPUT_SHAPE = INPUT_SHAPE[0]/NUM_POOL, INPUT_SHAPE[1]/NUM_POOL, 1
BATCH_SIZE = 300
TEST_BATCH_SIZE = 250
SHUFFLE_NUM = DIVIDEND * DIVIDEND * 200
EST_ITERATIONS = int(.8 * (DIVIDEND * DIVIDEND * 104000 // BATCH_SIZE))


def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [INPUT_SHAPE[0], INPUT_SHAPE[1], 3])
    return image / 255.0

def decode_label_coarse(label):
    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [OUTPUT_SHAPE[0], OUTPUT_SHAPE[1], 1])
    label = tf.cast(label, tf.float32)
    return label

def return_datatset_train():
    images = tf.data.FixedLengthRecordDataset(
      TRAIN_INPUT_SAVE, INPUT_SHAPE[0] * INPUT_SHAPE[1] * 3).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TRAIN_LABEL_SAVE_COARSE, OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]).map(decode_label_coarse)
    return tf.data.Dataset.zip((images, labels))

def return_datatset_test():
    images = tf.data.FixedLengthRecordDataset(
      TEST_INPUT_SAVE, INPUT_SHAPE[0] * INPUT_SHAPE[1] * 3).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TEST_LABEL_SAVE_COARSE, OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]).map(decode_label_coarse)
    return tf.data.Dataset.zip((images, labels))

def build_model(x, labels=None, reuse=False):
    if reuse:
        prefix = 'test_'
    else:
        prefix = 'train_'
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        conv_pointers = [InputLayer(x, name= 'c_disc_inputs')]
        for i,v in enumerate(CONVOLUTIONS):
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

            if i < len(CONVOLUTIONS)-1:
                # conv_pointers.append(ConcatLayer([curr_layer, conv_pointers[-1]],
                #  3, name = 'concat_layer%s'%(i)))
                conv_pointers.append(curr_layer)
            else:
                conv_pointers.append(curr_layer)
        # y_conv = DenseLayer(flat, m.fully_connected_size, act=tf.nn.relu,name =  'hidden_encode')
        max_pool = Conv2d(conv_pointers[-1],
            1, (1, 1),strides = (1,1), name='c_Final_Conv')
        # _, pm_width, pm_height, _ = pre_max_pool.outputs.get_shape()
        # max_pool_width, max_pool_height = pm_width, pm_height
        # max_pool = MaxPool2d(pre_max_pool, filter_size = (max_pool_width, max_pool_height), strides = (max_pool_width, max_pool_height), name = 'c_Final_Pool')
        if labels is None:
            return tf.round(tf.sigmoid(max_pool.outputs))
        logits = FlattenLayer(max_pool).outputs
        final_guess = tf.round(tf.sigmoid(logits))

        flat_labels = tf.contrib.layers.flatten(labels)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_labels, logits=logits))
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
        accuracy_summary = tf.summary.scalar(prefix + 'Accuracy', tf.reduce_mean(tf.cast(tf.equal(flat_labels, final_guess), tf.float32)))
        percent_found_summary_round = tf.summary.scalar(prefix + 'Percent Found Rounded', tf.reduce_mean(final_guess))
        percent_found_summary = tf.summary.scalar(prefix + 'Percent Found Nonrounded', tf.reduce_mean(tf.sigmoid(logits)))
        percent_real_summary = tf.summary.scalar(prefix + 'Percent Real', tf.reduce_mean(flat_labels))
        flat_labels = tf.cast(flat_labels, tf.float64)
        final_guess = tf.cast(final_guess, tf.float64)
        TP = tf.count_nonzero(final_guess * flat_labels, dtype=tf.float32)
        TN = tf.count_nonzero((final_guess - 1) * (flat_labels - 1), dtype=tf.float32)
        FP = tf.count_nonzero(final_guess * (flat_labels - 1), dtype=tf.float32)
        FN = tf.count_nonzero((final_guess - 1) * flat_labels, dtype=tf.float32)
        true_positive = tf.divide(TP, TP + FP)
        true_negative = tf.divide(TN, TN + FN)
        true_positive_summary =tf.summary.scalar(prefix + 'True Positive',true_positive)
        true_negative_summary =tf.summary.scalar(prefix + 'True Negative',true_negative)

        image_summary = tf.summary.image(prefix + "Example", tf.concat([tf.sigmoid(max_pool.outputs), labels, tf.ones_like(max_pool.outputs)], axis = 2),max_outputs = MAX_OUTPUTS)#show fake image

        cross_entropy_summary = tf.summary.scalar(prefix + 'Loss',cross_entropy)
        real_summary = tf.summary.merge([cross_entropy_summary,accuracy_summary,
            percent_found_summary_round,percent_found_summary, true_negative_summary,
            true_positive_summary, percent_real_summary])
        return real_summary, image_summary, train_step



if __name__ == "__main__":

    sess = tf.Session()#start the session
    ##############GET DATA###############
    train_ship = return_datatset_train().shuffle(SHUFFLE_NUM).repeat().batch(BATCH_SIZE)
    test_ship = return_datatset_test().shuffle(SHUFFLE_NUM).repeat().batch(TEST_BATCH_SIZE)
    train_iterator = train_ship.make_initializable_iterator()
    test_iterator = test_ship.make_initializable_iterator()
    train_input, train_label = train_iterator.get_next()
    test_input, test_label = test_iterator.get_next()
    sess.run([train_iterator.initializer, test_iterator.initializer])
    input_summary, image_summary,train_step = build_model(train_input, train_label)
    test_input_summary, test_image_summary,_ = build_model(test_input, test_label, reuse=True)

    ##########################################################################
    #Call function to make tf models
    ##########################################################################
    sess.run(tf.global_variables_initializer())
    #
    saver_perm = tf.train.Saver()
    if PERM_MODEL_FILEPATH is not None and RESTORE:
        saver_perm.restore(sess, PERM_MODEL_FILEPATH)
    else:
        print('SAVE')
        saver_perm.save(sess, PERM_MODEL_FILEPATH)


    train_writer = tf.summary.FileWriter(SUMMARY_FILEPATH,
                                  sess.graph)
    for i in range(ITERATIONS):
        if not i % WHEN_DISP:
            input_summary_ex, image_summary_ex, _= sess.run([input_summary,image_summary ,train_step])
            train_writer.add_summary(image_summary_ex, i)
            train_writer.add_summary(input_summary_ex, i)
        else:
            input_summary_ex, _= sess.run([input_summary, train_step])
            train_writer.add_summary(input_summary_ex, i)

        if not i % WHEN_TEST:
            if not i % WHEN_DISP:
                input_summary_ex, image_summary_ex= sess.run([test_input_summary, test_image_summary])
                train_writer.add_summary(image_summary_ex, i)
                train_writer.add_summary(input_summary_ex, i)
            else:
                input_summary_ex= sess.run(test_input_summary)
                train_writer.add_summary(input_summary_ex, i)

        if not i % WHEN_SAVE:
            saver_perm.save(sess, PERM_MODEL_FILEPATH)

        if not i % EST_ITERATIONS:
            print('Epoch' + str(i / EST_ITERATIONS))
