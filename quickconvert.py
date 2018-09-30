import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

COARSE_MODEL_FILEPATH2 = '/Models/Ships/CoarseShipsTrained/model.ckpt' #filepaths to model and summaries
FINE_MODEL_FILEPATH2 = '/Models/Ships/FineShipsValidated/model.ckpt' #filepaths to model and summaries
FINAL_FILEPATH = '/Models/Ships/FinalShipsDeep/model.ckpt'

coarse_old_vars = tf.contrib.framework.list_variables(COARSE_MODEL_FILEPATH2)
fine_old_vars = tf.contrib.framework.list_variables(FINE_MODEL_FILEPATH2)
with tf.Graph().as_default(), tf.Session().as_default() as sess:
    new_vars = []
    for name, shape in coarse_old_vars:
        v = tf.contrib.framework.load_variable(COARSE_MODEL_FILEPATH2, name)
        new_vars.append(tf.Variable(v, name=name))
    for name, shape in fine_old_vars:
        v = tf.contrib.framework.load_variable(FINE_MODEL_FILEPATH2, name)
        new_vars.append(tf.Variable(v, name=name))
    print(new_vars)
    saver = tf.train.Saver(new_vars)
    sess.run(tf.global_variables_initializer())
    saver.save(sess, FINAL_FILEPATH)
