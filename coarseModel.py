import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
import numpy as np

FILEPATH = '/Data/ShipDetection/'
DATA_FILEPATH = FILEPATH + 'train/'
CSV_FILENAME = DATA_FILEPATH + 'train_ship_segmentations.csv'
TRAIN_INPUT_SAVE = FILEPATH + 'coarse/'+ 'train_images'
TRAIN_LABEL_COARSE_SAVE = FILEPATH + 'coarse/' + 'train_labels_coarse'
PERM_MODEL_FILEPATH = '/Models/Ships/CoarseShipsTrained/model.ckpt' #filepaths to model and summaries
SUMMARY_FILEPATH ='/Models/Ships/CoarseShipsTrained/Summaries/'

RESTORE = False
WHEN_DISP = 10
WHEN_SAVE = 2000
MAX_OUTPUTS = 16
ITERATIONS = 1000000
LEARNING_RATE = 3e-3

CONVOLUTIONS = [64, 128, 256, 512]
DIVIDEND = 24
BASE_INPUT_SHAPE = 768, 768, 3
BASE_OUTPUT_SHAPE = 768, 768, 1
COURSE_SHAPE = DIVIDEND, DIVIDEND
INPUT_SHAPE = BASE_INPUT_SHAPE[0]/DIVIDEND, BASE_INPUT_SHAPE[1]/DIVIDEND, 3
OUTPUT_SHAPE = BASE_OUTPUT_SHAPE[0]/DIVIDEND, BASE_OUTPUT_SHAPE[1]/DIVIDEND, 1
BATCH_SIZE = 16

def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [BASE_INPUT_SHAPE[0], BASE_INPUT_SHAPE[1], 3])
    return image / 255.0

def decode_label_coarse(label):
    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [COURSE_SHAPE[0], COURSE_SHAPE[1]])
    label = tf.cast(label, tf.float32)
    return label

def decode_label(label):
    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [COURSE_SHAPE[0], COURSE_SHAPE[1]])
    label = tf.cast(label, tf.float32)
    condition = tf.equal(label, 0)
    case_true = tf.zeros_like(label)+ .1
    # case_true = tf.reshape(tf.multiply(tf.ones([8], tf.int32), -9999), [2, 4])
    return  tf.where(condition, case_true, label)

def return_datatset_train():
    images = tf.data.FixedLengthRecordDataset(
      TRAIN_INPUT_SAVE, BASE_INPUT_SHAPE[0] * BASE_INPUT_SHAPE[1] * 3).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TRAIN_LABEL_COARSE_SAVE, COURSE_SHAPE[0]* COURSE_SHAPE[1]).map(decode_label_coarse)
    return tf.data.Dataset.zip((images, labels))

def build_model(x, labels):
    conv_pointers = [InputLayer(x, name= 'c_disc_inputs')]
    for i,v in enumerate(CONVOLUTIONS):
        curr_layer = BatchNormLayer(Conv2d(conv_pointers[-1],
            CONVOLUTIONS[i], (5, 5),strides = (2,2), name=
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
    pre_max_pool = Conv2d(conv_pointers[-1],
        1, (5, 5),strides = (1,1), name='c_Final_Conv')
    _, pm_width, pm_height, _ = pre_max_pool.outputs.get_shape()
    max_pool_width, max_pool_height = pm_width/DIVIDEND, pm_height/DIVIDEND
    max_pool = MaxPool2d(pre_max_pool, filter_size = (max_pool_width, max_pool_height), strides = (max_pool_width, max_pool_height), name = 'c_Final_Pool')
    logits = FlattenLayer(max_pool).outputs
    flat_labels = tf.contrib.layers.flatten(labels)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_labels, logits=logits))
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    final_guess = tf.round(tf.sigmoid(logits))



    accuracy_summary = tf.summary.scalar('Accuracy', tf.reduce_mean(tf.cast(tf.equal(flat_labels, final_guess), tf.float32)))
    percent_found_summary_round = tf.summary.scalar('Percent Found Rounded', tf.reduce_mean(final_guess))
    percent_found_summary = tf.summary.scalar('Percent Found Nonrounded', tf.reduce_mean(tf.sigmoid(logits)))

    flat_labels = tf.cast(flat_labels, tf.float64)
    final_guess = tf.cast(final_guess, tf.float64)
    TP = tf.count_nonzero(final_guess * flat_labels, dtype=tf.float32)
    TN = tf.count_nonzero((final_guess - 1) * (flat_labels - 1), dtype=tf.float32)
    FP = tf.count_nonzero(final_guess * (flat_labels - 1), dtype=tf.float32)
    FN = tf.count_nonzero((final_guess - 1) * flat_labels, dtype=tf.float32)
    true_positive = tf.divide(TP, TP + FP)
    true_negative = tf.divide(TN, TN + FN)
    # accuracy = tf.divide(TP + TN, TN + FN + TP + FP)
    # accuracy_summary = tf.summary.scalar('Accuracy', accuracy)
    true_positive_summary =tf.summary.scalar('True Positive',true_positive)
    true_negative_summary =tf.summary.scalar('True Negative',true_negative)
    # resized_label = tf.image.resize_images(labels,size = (BASE_INPUT_SHAPE[0], BASE_INPUT_SHAPE[1]))
    # resized_output = tf.image.resize_images(tf.sigmoid(max_pool.outputs),size = (BASE_INPUT_SHAPE[0], BASE_INPUT_SHAPE[1]))
    # tiled_labels = tf.tile(tf.expand_dims(resized_label, axis = -1), [1,1,1,3])
    # tiled_outputs = tf.tile(resized_output, [1,1,1,3])

    image_summary = tf.summary.image("Example", tf.concat([tf.sigmoid(max_pool.outputs), tf.expand_dims(labels, axis = -1), tf.ones_like(max_pool.outputs)], axis = 2),max_outputs = MAX_OUTPUTS)#show fake image
    # image_summary_2 = tf.summary.image("Example_2", x,max_outputs = MAX_OUTPUTS)#show fake image
    # image_summary_merge = tf.summary.merge([image_summary,image_summary_2])

    cross_entropy_summary = tf.summary.scalar('Loss',cross_entropy)
    real_summary = tf.summary.merge([cross_entropy_summary,accuracy_summary,
        percent_found_summary_round,percent_found_summary, true_negative_summary,
        true_positive_summary])
    return real_summary , image_summary,train_step



if __name__ == "__main__":

    sess = tf.Session()#start the session
    ##############GET DATA###############
    train_ship = return_datatset_train().repeat().batch(BATCH_SIZE)
    # test_ship = getData.return_mnist_dataset_test().repeat().batch(TEST_BATCH_SIZE)
    train_iterator = train_ship.make_initializable_iterator()
    # test_iterator = test_mnist.make_initializable_iterator()
    train_input, train_label = train_iterator.get_next()
    # test_input = test_iterator.get_next()
    sess.run([train_iterator.initializer])
    input_summary, image_summary,train_step = build_model(train_input, train_label)

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

        if not i % WHEN_SAVE:
            saver_perm.save(sess, PERM_MODEL_FILEPATH)

    train_model(head_block , ITERATIONS, test_bool)
