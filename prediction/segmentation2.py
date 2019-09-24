import tensorflow as tf
import numpy as np
import cv2 as cv
from keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
import os

'''
调用tensorflow语义分割，将人物提取出来
'''

img_path=os.getcwd()+'/imgs/mini_zjz.png'
model_path=os.getcwd()+'/models/maskmode/frozen_inference_graph.pb'
img = load_img(img_path)  # 输入预测图片的url

# img = load_img('datasets/testset/JPEGImages/2018_010001.jpg')
# img = load_img(
#     '/home/pzn/anaconda3/envs/DeepLabV3/lib/python3.6/site-packages/tensorflow/models/research/deeplab/datasets/testset/JPEGImages/2018_010001.jpg')
img = img_to_array(img)
img = np.expand_dims(img, axis=0).astype(np.uint8)  # uint8是之前导出模型时定义的

# # 加载模型
# sess = tf.Session()
# #with open("output_model/frozen_inference_graph_473.pb", "rb") as f:
# with open("/home/pzn/anaconda3/envs/DeepLabV3/lib/python3.6/site-packages/tensorflow/models/research/deeplab/output_model/frozen_inference_graph_0325.pb", "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     '''
#     output = tf.import_graph_def(graph_def, input_map={"ImageTensor:0": img},
#                                      return_elements=["SemanticPredictions:0"])
#     # input_map 就是指明 输入是什么；
#     # return_elements 就是指明输出是什么；两者在前面已介绍
#     '''

#     output = tf.import_graph_def(graph_def, input_map={"ImageTensor:0": img},
#                                      return_elements=["SemanticPredictions:0"])

graph = tf.Graph()
INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
graph_def = None
# graph_path = "/home/pzn/anaconda3/envs/DeepLabV3/lib/python3.6/site-packages/tensorflow/models/research/deeplab/output_model/frozen_inference_graph_0325.pb"
with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

if graph_def is None:
    raise RuntimeError('Cannot find inference graph in tar archive.')

with graph.as_default():
    tf.import_graph_def(graph_def, name='')

sess = tf.Session(graph=graph)
result = sess.run(
    OUTPUT_TENSOR_NAME,
    feed_dict={INPUT_TENSOR_NAME: img})

# result = sess.run(output)
print(type(result))
# result.save('aaa.png')
# cv.imshow(result[0])
print(result[0].shape)  # (1, height, width)

print(result)
print(result[0])
# result[0].save('aaa.png')
# img_ = img[:,:,::-1].transpose((2,0,1))

cv.imwrite('aaa.jpg', result.transpose((1, 2, 0)))
plt.imshow(result[0])
plt.show()