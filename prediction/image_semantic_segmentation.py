import os
import tensorflow as tf
import sys
import  numpy as np
from matplotlib import pyplot as plt
from skimage import  transform
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import cv2
'''
加载pb模型实现语义分割,图片大小要在513内
'''

model_dir=os.getcwd()+'/models/maskmode'
_IMAGE_SIZE = 64
def standardize(img):
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    return img

def load_image( infilename ) :
    img = Image.open( infilename )
    img = img.resize((_IMAGE_SIZE, _IMAGE_SIZE))
    img.load()
    data = np.asarray( img, dtype=np.float32 )
    data = standardize(data)
    return data


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      model_dir, 'frozen_inference_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  imgRes = cv2.imread(image_path)
  rows, cols,_ = imgRes.shape
  print(imgRes.shape)


  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()
  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    inputs = sess.graph.get_tensor_by_name("ImageTensor:0")
    probabilities_op = sess.graph.get_tensor_by_name('SemanticPredictions:0')
    class_index_op = sess.graph.get_tensor_by_name('ArgMax:0')

    # image_data = load_image(image_path)
    img = load_img(image_path)  # 输入预测图片的url
    img = img_to_array(img)
    image_data = np.expand_dims(img, axis=0).astype(np.uint8)  # uint8是之前导出模型时定义的
    print(image_data.shape)
    # print(image_data)
    result = sess.run(probabilities_op,
                                          feed_dict={inputs: image_data})
    n=result.transpose(1,2,0)#result是一个（1，xx,xx）的shape，因为我们要用的是一个通道在后（xx，xx，1）这样的shape
    print(n.shape)
    print(result[0].shape)
    img = np.array(n, dtype=np.uint8)#转换类型

    # 遍历替换
    for i in range(rows):
        for j in range(cols):
            # print("[{0},{1}]:{2}".format(i,j,img[i,j]))
            # print('------')
            if img[i, j] == 0:
                imgRes[i, j] = (255, 255, 255)  # 此处替换颜色为bgr通道

    cv2.imshow('img',imgRes)
    cv2.waitKey(0)

if __name__ == '__main__':
    argv = sys.argv
    if(len(argv) < 2):
        print("usage: python nsfw_predict <image_path>")
    image_path = argv[1]
    print(image_path)
    run_inference_on_image(image_path)