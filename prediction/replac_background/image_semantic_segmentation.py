import os
import tensorflow as tf
import sys
import  numpy as np
import time
from matplotlib import pyplot as plt
from skimage import  transform
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import cv2

import argparse

sys.path.append(os.getcwd()+'/prediction/replac_background/deep_image_matting')

import net
import torch
from deploy import inference_img_whole
'''
加载pb模型实现语义分割,图片大小要在513内
'''
isDebug=False
model_dir=os.getcwd()+'/models/maskmode'
_IMAGE_SIZE = 64
'''
生成trimap图
'''
def generate_trimap(alpha):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))

    fg = cv2.erode(fg, kernel, iterations=10)#腐蚀为了让未知区域可以大一点
    if isDebug:
        cv2.imshow('after', fg)
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv2.dilate(unknown, kernel, iterations=10)
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)

def custom_blur(image):
  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
  dst = cv2.filter2D(image, -1, kernel=kernel)
  if isDebug:
    cv2.imshow("custom_blur_demo", dst)
  return dst

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      model_dir, 'frozen_inference_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

'''
将前景扣出并与新背景合并生成新图
'''
def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y:y + h, x:x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    if isDebug:
        cv2.imshow('im',im)
    return im, bg

def run_inference_on_image(image,isDebug = False):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  global image_path
  image_path = image
  imgRes = cv2.imread(image)
  print(imgRes.shape)
  imgRes=resizeImg(imgRes)
  rows, cols = imgRes.shape[:2]

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
    # class_index_op = sess.graph.get_tensor_by_name('ArgMax:0')

    # image_data = load_image(image_path)
    # img = load_img(image_path)  # 输入预测图片的url
    # img = img_to_array(img)
    # image_data = np.expand_dims(img, axis=0).astype(np.uint8)  # uint8是之前导出模型时定义的
    # print(image_data.shape)

    image_newdata=cv2.cvtColor(imgRes,cv2.COLOR_BGR2RGB)#必须转为rbg，因为模型训练时用的是rgb的图，否则会特征不匹配
    image_newdata = np.expand_dims(image_newdata, axis=0).astype(np.uint8)
    # print(image_data)
    result = sess.run(probabilities_op,
                                          feed_dict={inputs: image_newdata})
    n=result.transpose(1,2,0)#result是一个（1，xx,xx）的shape，因为我们要用的是一个通道在后（xx，xx，1）这样的shape
    print(n.shape)
    print(result[0].shape)
    img = np.array(n, dtype=np.uint8)#转换类型
    if isDebug:
        cv2.imshow('mark', img)

    #人的区域改为白色
    copymask = np.copy(img)
    for i in range(rows):
        for j in range(cols):
            if img[i, j] != 0:
                copymask[i, j] = 255
    # cv2.imshow('copymask',copymask)
    print('copymask')
    print(np.asarray(copymask))
    row,col,_=copymask.shape
    remask=np.reshape(copymask, [row, col])#后面要用的是一个这样的
    print(remask.shape)

    alpha = np.zeros((rows, cols), np.float32)
    alpha[0:rows, 0:cols] = remask
    trimap = generate_trimap(alpha)
    if isDebug:
        cv2.imshow('trimap',trimap)
    print('trimap')
    print(trimap.shape)
    result=trimap[:, :, np.newaxis]
    timestamp=str(int(time.time()))
    cv2.imwrite(os.getcwd()+'/trimaps/trimap'+timestamp+'.png',result)
    #开始通过deep-image-matting处理
    path=predictImg(imgRes,cv2.imread(os.getcwd()+'/trimaps/trimap'+timestamp+'.png')[:, :, 0],row,col)
    if isDebug:
        cv2.waitKey(0)
    return path

INPUT_SIZE=513
'''
由于模型最大支持的图片尺寸为513,所以要进行缩放处理
'''
def resizeImg(inputImg):
    row,col=inputImg.shape[:2]
    #如果比规定的值大那必须要进行处理
    if row>INPUT_SIZE or col>INPUT_SIZE:
        fx=(float)(INPUT_SIZE/row)
        fy=(float)(INPUT_SIZE/col)
        resized=cv2.resize(inputImg,None,fx=min(fx,fy),fy=min(fx,fy),interpolation=cv2.INTER_LANCZOS4)
        print('resized')
        print(resized.shape)
        return resized
    else:
        return inputImg


def predictImg(image,trimap,row,col):
    result_dir = os.getcwd()+"/pred"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # parameters setting
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cuda = False
    args.resume = "models/deep_image_matting/stage_sad.pth"
    args.stage = 1
    args.crop_or_resize = "whole"
    args.max_size = 1600

    # init model
    model = net.VGG16(args)
    ckpt = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=True)
    # model = model.cuda()

    # infer one by one
    torch.cuda.empty_cache()
    with torch.no_grad():
        pred_mattes = inference_img_whole(args, model, image, trimap)

    pred_mattes = (pred_mattes * 255).astype(np.uint8)
    pred_mattes[trimap == 255] = 255
    pred_mattes[trimap == 0] = 0
    # target = image * pred_mattes[:,:,np.newaxis]
    # cv2.imshow('target', cv2.cvtColor(target,cv2.COLOR_BGR2RGB))
    # cv2.waitKey(0)
    # bg = cv2.imread(os.getcwd() + '/composebg/input_image.jpg')
    bg = create_bgimage(row,col)
    # print('00000000')
    # cv2.imshow('img33',image)
    im, bg = composite4(image, bg, pred_mattes, col, row)
    timestamp = str(int(time.time()))
    cv2.imwrite(os.getcwd()+'/pred/test'+timestamp+'.png', pred_mattes)
    fileNameAndSuffix=os.path.split(image_path)[1].split('.')
    newfileNameAndSuffix=os.getcwd()+'/pred/'+fileNameAndSuffix[0]+'_replace_bg'+timestamp+'.'+fileNameAndSuffix[1]
    print('=============:'+newfileNameAndSuffix)
    cv2.imwrite(newfileNameAndSuffix,im)
    if isDebug:
        cv2.waitKey(0)
    return newfileNameAndSuffix
'''
创建纯色图
https://blog.csdn.net/qq_42444944/article/details/90745397
'''
def create_bgimage(row,col):
    img = np.zeros([row, col, 3], np.uint8)
    img[:, :, 0] = np.zeros([row, col]) + 205
    img[:, :, 1] = np.ones([row, col]) + 105
    img[:, :, 2] = np.ones([row, col]) * 63
    # cv2.imshow("iamge", img)
    # img2 = np.zeros([row, col, 3], np.uint8)+233
    # cv2.imshow("iamge2", img2)
    # cv2.waitKey(0)
    return img

if __name__ == '__main__':
    # argv = sys.argv
    # if(len(argv) < 2):
    #     print("usage: python nsfw_predict <image_path>")
    # image_path = argv[1]
    image_path='imgs/white_zjz.jpeg'
    print(image_path)
    run_inference_on_image(image_path,True)