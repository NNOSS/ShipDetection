import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
import numpy as np

FILEPATH = '/Data/ShipDetection/'
CSV_FILENAME = FILEPATH + 'train_ship_segmentations.csv'
DATA_FILEPATH = FILEPATH + 'train/'
TRAIN_INPUT_SAVE = FILEPATH + 'train_images_6'
TRAIN_LABEL_SAVE = FILEPATH + 'train_labels_6'
PERM_MODEL_FILEPATH = '/Models/Ships/model.ckpt' #filepaths to model and summaries
SUMMARY_FILEPATH ='/Models/Ships/Summaries/'

RESTORE = False
WHEN_DISP = 100
WHEN_SAVE = 2000
MAX_OUTPUTS = 32
ITERATIONS = 1000000
LEARNING_RATE = 3e-3

CONVOLUTIONS = [32, 64, 128,256]
DIVIDEND = 6
BASE_INPUT_SHAPE = 768, 768, 3
BASE_OUTPUT_SHAPE = 768, 768, 1
INPUT_SHAPE = BASE_INPUT_SHAPE[0]/DIVIDEND, BASE_INPUT_SHAPE[1]/DIVIDEND, 3
OUTPUT_SHAPE = BASE_OUTPUT_SHAPE[0]/DIVIDEND, BASE_OUTPUT_SHAPE[1]/DIVIDEND, 1
BATCH_SIZE = 32

def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [INPUT_SHAPE[0], INPUT_SHAPE[1], 3])
    return image / 255.0

def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]])
    label = tf.cast(label, tf.float32)
    condition = tf.equal(label, 0)
    case_true = tf.zeros_like(label)+ .1
    # case_true = tf.reshape(tf.multiply(tf.ones([8], tf.int32), -9999), [2, 4])
    return  tf.where(condition, case_true, label)

def return_datatset_train():
    images = tf.data.FixedLengthRecordDataset(
      TRAIN_INPUT_SAVE, INPUT_SHAPE[0] * INPUT_SHAPE[1] * 3).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TRAIN_LABEL_SAVE, OUTPUT_SHAPE[0]* OUTPUT_SHAPE[1]).map(decode_label)
    return tf.data.Dataset.zip((images, labels))

def build_model(x, labels):
    conv_pointers = [InputLayer(x, name= 'disc_inputs')]
    for i,v in enumerate(CONVOLUTIONS):
        curr_layer = BatchNormLayer(Conv2d(Conv2d(conv_pointers[-1],
            CONVOLUTIONS[i], (1, 1),strides = (1,1), name=
            '1x1_conv1_%s'%(i)),
            CONVOLUTIONS[i], (5, 5),strides = (1,1), name=
            'conv1_%s'%(i)),
            act=tf.nn.leaky_relu,is_train=True ,name=
            'batch_norm%s'%(i))




        if i < len(CONVOLUTIONS)-1:
            conv_pointers.append(ConcatLayer([curr_layer, conv_pointers[-1]],
             3, name = 'concat_layer%s'%(i)))
        else:
            conv_pointers.append(curr_layer)
    # y_conv = DenseLayer(flat, m.fully_connected_size, act=tf.nn.relu,name =  'hidden_encode')
    output = BatchNormLayer(Conv2d(Conv2d(conv_pointers[-1],
        CONVOLUTIONS[i], (1, 1),strides = (1,1), name='1x1_Final_Conv'),
        1, (5, 5),strides = (1,1), name='Final_Conv'),is_train=True ,name='Final_Norm')

    logits = FlattenLayer(output).outputs
    flat_labels = tf.contrib.layers.flatten(labels)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_labels, logits=logits))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    tiled_labels = tf.tile(tf.expand_dims(labels, axis = -1), [1,1,1,3])
    tiled_outputs = tf.tile(tf.sigmoid(output.outputs), [1,1,1,3])
    input_summary = tf.summary.image("Example", tf.concat([x, tiled_labels, tiled_outputs], axis = 2),max_outputs = MAX_OUTPUTS)#show fake image
    cross_entropy_summary = tf.summary.scalar('Loss',cross_entropy)
    # real_summary = tf.summary.merge([input_summary,cross_entropy_summary])
    return input_summary, cross_entropy_summary , train_step



if __name__ == "__main__":

    sess = tf.Session()#start the session
    ##############GET DATA###############
    train_ship = return_datatset_train().shuffle(buffer_size= 5000).repeat().batch(BATCH_SIZE)
    # test_ship = getData.return_mnist_dataset_test().repeat().batch(TEST_BATCH_SIZE)
    train_iterator = train_ship.make_initializable_iterator()
    # test_iterator = test_mnist.make_initializable_iterator()
    train_input, train_label = train_iterator.get_next()
    # test_input = test_iterator.get_next()
    sess.run([train_iterator.initializer])
    input_summary, cross_entropy_summary, train_step = build_model(train_input, train_label)

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
        input_summary_ex, cross_entropy_summary_ex, _= sess.run([input_summary, cross_entropy_summary, train_step])
        train_writer.add_summary(cross_entropy_summary_ex, i)

        if not i % WHEN_DISP:
            train_writer.add_summary(input_summary_ex, i)
        if not i % WHEN_SAVE:
            saver_perm.save(sess, PERM_MODEL_FILEPATH)

    # train_model(head_block , ITERATIONS, test_bool)
