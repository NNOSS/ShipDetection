import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
import numpy as np
import importImg
import saveMovie
import imageio

FILEPATH = '/Data/ShipDetection/'
CSV_FILENAME = FILEPATH + 'train_ship_segmentations.csv'
DATA_FILEPATH = FILEPATH + 'train/'
TRAIN_INPUT_SAVE = FILEPATH + 'train_images'
TRAIN_LABEL_SAVE = FILEPATH + 'train_labels'
PERM_MODEL_FILEPATH = '/home/gtower/Models/Ships/model.ckpt' #filepaths to model and summaries
RESTORE = False

CONVOLUTIONS = [64, 128, 256]
INPUT_SHAPE = 768, 768, 3
OUTPUT_SHAPE = 768, 768, 1
BATCH_SIZE = 1000

def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [INPUT_SHAPE[0], INPUT_SHAPE[1], 3])
    return image / 255.0

def decode_label(label):
    label = tf.decode_raw(label, tf.bool)
    label = tf.reshape(label, [OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]])
    return label

def return_datatset_train():
    images = tf.data.FixedLengthRecordDataset(
      TRAIN_INPUT_SAVE, INPUT_SHAPE[0] * INPUT_SHAPE[1] * 3).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TRAIN_LABEL_SAVE, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]).map(decode_label)
    return tf.data.Dataset.zip((images, labels))


if __name__ == "__main__":

    sess = tf.Session()#start the session
    ##############GET DATA###############
    train_ship = getData.return_datatset_train().repeat().batch(BATCH_SIZE)
    # test_ship = getData.return_mnist_dataset_test().repeat().batch(TEST_BATCH_SIZE)
    train_iterator = train_mnist.make_initializable_iterator()
    # test_iterator = test_mnist.make_initializable_iterator()
    train_input = train_iterator.get_next()
    # test_input = test_iterator.get_next()
    sess.run([train_iterator.initializer])
    ##########################################################################
    #Call function to make tf models
    ##########################################################################
    saver_perm = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if PERM_MODEL_FILEPATH is not None and RESTORE:
        saver_perm.restore(sess, PERM_MODEL_FILEPATH)
    else:
        print('SAVE')
        saver_perm.save(sess, PERM_MODEL_FILEPATH)

    train_writer = tf.summary.FileWriter(SUMMARY_FILEPATH,
                                  sess.graph)
    train_model(head_block , ITERATIONS, test_bool)
