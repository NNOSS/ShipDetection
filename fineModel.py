import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
import numpy as np

FILEPATH = '/Data/ShipDetection/'
DATA_FILEPATH = FILEPATH + 'train/'
TRAIN_LABEL_SAVE = FILEPATH + 'fine/' +'train_labels_fine'
TRAIN_INPUT_SAVE = FILEPATH + 'fine/' +'train_images_fine'
TEST_LABEL_SAVE = FILEPATH + 'fine/' +'test_labels_fine'
TEST_INPUT_SAVE = FILEPATH + 'fine/' +'test_images_fine'
PERM_MODEL_FILEPATH = '/Models/Ships/FineShipsValidated/model.ckpt' #filepaths to model and summaries
SUMMARY_FILEPATH ='/Models/Ships/FineShipsValidated/Summaries/'

RESTORE = True
WHEN_DISP = 100
WHEN_SAVE = 2000
WHEN_TEST = 50
MAX_OUTPUTS = 16
ITERATIONS = 1000000
LEARNING_RATE = 3e-3

EST_ITERATIONS = 104000 // 64
CONVOLUTIONS = [32, -32, 64, -64, 128, -128]
DECONVOLUTIONS = [-128, 128, -64, 64, -32, 32]
DIVIDEND = 24
BASE_INPUT_SHAPE = 768, 768, 3
BASE_OUTPUT_SHAPE = 768, 768, 1
INPUT_SHAPE = BASE_INPUT_SHAPE[0]/DIVIDEND, BASE_INPUT_SHAPE[1]/DIVIDEND, 3
OUTPUT_SHAPE = BASE_OUTPUT_SHAPE[0]/DIVIDEND, BASE_OUTPUT_SHAPE[1]/DIVIDEND, 1
BATCH_SIZE = 64
TEST_BATCH_SIZE = 500

def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [INPUT_SHAPE[0], INPUT_SHAPE[1], 3])
    return image / 255.0

def decode_label_fine(label):
    label = tf.decode_raw(label, tf.uint8)
    label = tf.reshape(label, [OUTPUT_SHAPE[0], OUTPUT_SHAPE[1]])
    label = tf.cast(label, tf.float32)
    return label

def return_datatset_train():
    images = tf.data.FixedLengthRecordDataset(
      TRAIN_INPUT_SAVE, INPUT_SHAPE[0] * INPUT_SHAPE[1] * 3).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TRAIN_LABEL_SAVE, OUTPUT_SHAPE[0]* OUTPUT_SHAPE[1]).map(decode_label_fine)
    return tf.data.Dataset.zip((images, labels))

def return_datatset_test():
    images = tf.data.FixedLengthRecordDataset(
      TEST_INPUT_SAVE, INPUT_SHAPE[0] * INPUT_SHAPE[1] * 3).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
      TEST_LABEL_SAVE, OUTPUT_SHAPE[0]* OUTPUT_SHAPE[1]).map(decode_label_fine)
    return tf.data.Dataset.zip((images, labels))

def build_model(x, labels, reuse = False):

    if reuse:
        prefix = 'test_'
    else:
        prefix = 'train_'
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        conv_pointers = [InputLayer(x, name= 'f_disc_inputs')]
        conv_pointers_concat = []
        for i,v in enumerate(CONVOLUTIONS):
            if v < 0:
                v *= -1
                curr_layer = Conv2d(BatchNormLayer(conv_pointers[-1],
                    act=tf.nn.relu,is_train=True ,name=
                    'f_batch_norm%s'%(i)),
                    v, (5, 5),strides = (1,1), name=
                    'f_conv1_%s'%(i))
                conv_pointers_concat.append(curr_layer)
                curr_layer = MaxPool2d(curr_layer, filter_size = (2,2), strides = (2,2), name = 'f_pool_%s'%(i))
                conv_pointers.append(curr_layer)
            else:
                curr_layer = Conv2d(BatchNormLayer(conv_pointers[-1],
                act=tf.nn.relu,is_train=True ,name=
                'f_batch_norm%s'%(i)),
                    v, (5, 5),strides = (1,1), name=
                    'f_conv1_%s'%(i))
                conv_pointers.append(curr_layer)

        deconv_pointers = [conv_pointers[-1]]
        for i,v in enumerate(DECONVOLUTIONS):
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
        logits = FlattenLayer(final_layer).outputs
        flat_labels = tf.contrib.layers.flatten(labels)
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=flat_labels, logits=logits))
        if reuse:
            train_step = None
        else:
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
        final_guess = tf.round(tf.sigmoid(logits))
        accuracy_summary = tf.summary.scalar(prefix + 'Accuracy', tf.reduce_mean(tf.cast(tf.equal(flat_labels, final_guess), tf.float32)))
        percent_found_summary_round = tf.summary.scalar(prefix + 'Percent Found Rounded', tf.reduce_mean(final_guess))
        percent_found_summary = tf.summary.scalar(prefix + 'Percent Found Nonrounded', tf.reduce_mean(tf.sigmoid(logits)))

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
        tiled_labels = tf.tile(tf.expand_dims(labels, axis = -1), [1,1,1,3])
        tiled_outputs = tf.tile(tf.sigmoid(final_layer.outputs), [1,1,1,3])

        image_summary = tf.summary.image(prefix + "Example", tf.concat([x, tiled_outputs, tiled_labels], axis = 2),max_outputs = MAX_OUTPUTS)#show fake image
        # image_summary_2 = tf.summary.image("Example_2", x,max_outputs = MAX_OUTPUTS)#show fake image
        # image_summary_merge = tf.summary.merge([image_summary,image_summary_2])

        cross_entropy_summary = tf.summary.scalar(prefix + 'Loss',cross_entropy)
        real_summary = tf.summary.merge([cross_entropy_summary,accuracy_summary,
            percent_found_summary_round,percent_found_summary, true_negative_summary,
            true_positive_summary])
        return real_summary , image_summary,train_step



if __name__ == "__main__":

    sess = tf.Session()#start the session
    ##############GET DATA###############
    train_ship = return_datatset_train().repeat().batch(BATCH_SIZE)
    test_ship = return_datatset_test().repeat().batch(TEST_BATCH_SIZE)


    # test_ship = getData.return_mnist_dataset_test().repeat().batch(TEST_BATCH_SIZE)
    train_iterator = train_ship.make_initializable_iterator()
    test_iterator = test_ship.make_initializable_iterator()
    train_input, train_label = train_iterator.get_next()
    test_input, test_label = test_iterator.get_next()
    sess.run([train_iterator.initializer, test_iterator.initializer])
    input_summary, image_summary,train_step = build_model(train_input, train_label)
    test_input_summary, test_image_summary,_ = build_model(test_input, test_label,reuse=True)


    ##########################################################################
    #Call function to make tf models
    ##########################################################################
    sess.run(tf.global_variables_initializer())

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

    train_model(head_block , ITERATIONS, test_bool)
