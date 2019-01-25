
# Show checkpoints
#import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
#slim = tf.contrib.slim

import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets 


print('\n vgg:')
reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/vgg/ssd_300_vgg.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)

print('\n\n mobilenetV1 :')

reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/mobilenet_v1/mobilenet_v1_1.0_224.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)

print('\n\n ssd_mobilenetV1 :')

reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/ssd_mobilenet_v1/model.ckpt-10518')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)

print('\n\n mobilenetV2 :')

reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/mobilenet_v2/mobilenet_v2_1.4_224.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)


#print([var.name for var in saver._var_list])

'''
with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm,
     normalizer_params={'is_training': False, 'updates_collections': None}):
     inception,inception_layers=slim.nets.inception.inception_v3_base(tf_images)
for v in sorted(inception_layers):
   print(v)
'''
    
'''
with slim.arg_scope(slim.nets.inception.inception_v3_arg_scope()):
            logits, endpoints = slim.nets.inception.inception_v3(input, num_classes=1001, is_training=False)
for v in sorted(endpoints):
   print(v)
'''


#vgg =  slim.arg_scope(slim.nets.vgg)

   
