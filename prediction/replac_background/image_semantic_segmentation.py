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

def generate_trimap(alpha):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = np.array(np.equal(alpha, 255).astype(np.float32))
    fg = cv2.erode(fg, kernel, iterations=np.random.randint(3, 6))
    unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
    unknown = cv2.dilate(unknown, kernel, iterations=np.random.randint(10, 20))
    trimap = fg * 255 + (unknown - fg) * 128
    return trimap.astype(np.uint8)

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      model_dir, 'frozen_inference_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

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
    cv2.imshow('im',im)
    cv2.imshow('bg',bg )
    return im, bg

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
    cv2.imshow('mark', img)

    #人的区域改为白色
    copymask = np.copy(img)
    for i in range(rows):
        for j in range(cols):
            if img[i, j] != 0:
                copymask[i, j] = 255
    cv2.imshow('copymask',copymask)
    print('copymask')
    print(np.asarray(copymask))

    remask=np.reshape(copymask, [500, 374])
    print(remask.shape)

    gimg = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    print('gimg')
    print(gimg)

    alpha = np.zeros((rows, cols), np.float32)
    alpha[0:rows, 0:cols] = remask
    trimap = generate_trimap(alpha)
    cv2.imshow('trimap',trimap)
    print('trimap')
    print(trimap.shape)
    result=trimap[:, :, np.newaxis]
    cv2.imwrite('trimap.png',result)



    #重新载入原图，得到hsv图后面要根据mask得出背景图的范围
    resImg=cv2.imread(image_path)
    hsv = cv2.cvtColor(resImg, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv)
    lowerArr=hsv[0,0]
    upperArr=hsv[0,cols-1]
    # 遍历替换
    for i in range(rows):
        for j in range(cols):
            # print("[{0},{1}]:{2}".format(i,j,img[i,j]))
            # print('------')
            if img[i, j] == 0:
                #将mask区中的最小值和最大值找出重新进行mask找到背景以便自动滤除
                pix=hsv[i,j]
                # print(pix)

                lowerArr[0]=lowerArr[0] if lowerArr[0]<pix[0] else pix[0]
                lowerArr[1] = lowerArr[1] if lowerArr[1] < pix[1] else pix[1]
                lowerArr[2] = lowerArr[2] if lowerArr[2] < pix[2] else pix[2]

                upperArr[0] = upperArr[0] if upperArr[0] > pix[0] else pix[0]
                upperArr[1] = upperArr[1] if upperArr[1] > pix[1] else pix[1]
                upperArr[2] = upperArr[2] if upperArr[2] > pix[2] else pix[2]


                # 替换背景颜色为bgr通道白色
                imgRes[i, j] = (255, 255, 255)
    print(lowerArr)
    print(upperArr)
    print(np.array([lowerArr[0]+20,lowerArr[1]+20,lowerArr[2]+20]))
    print(np.array([upperArr[0],upperArr[1],upperArr[2]]))
    cv2.imshow('img',imgRes)
    #开始重新处理边缘
    mask2 = cv2.inRange(hsv, np.array([lowerArr[0]+20,lowerArr[1]+20,lowerArr[2]+20]), np.array([upperArr[0]+0,upperArr[1]+0,upperArr[2]+0]))
    cv2.imshow('mask2',mask2)




    # kernel2 = np.ones((30, 30), np.uint8)
    # dilate = cv2.dilate(mask2, kernel2)
    # cv2.imshow('dilate', dilate)
    # erode=cv2.erode(dilate,kernel2)
    # cv2.imshow('erode',erode)


    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
    cv2.imshow('opening', opening)
    # 取轮廓，大于61的值应变为122，小于61的应变为0，THRESH_BINARY_INV则相反
    _, thresh = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 200, 220)
    cv2.imshow('edges', edges)

    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
    cv2.putText(img, "{:.3f}".format(len(contours)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
    # cv2.imshow("img", img)
    cv2.imshow("contours", img)


    # 遍历替换
    for i in range(rows):
        for j in range(cols):
            if opening[i, j] == 255:
                imgRes[i, j] = (255, 255, 255)  # 此处替换颜色为bgr通道
    cv2.imshow('img2', imgRes)
    cv2.waitKey(0)

if __name__ == '__main__':
    argv = sys.argv
    if(len(argv) < 2):
        print("usage: python nsfw_predict <image_path>")
    image_path = argv[1]
    print(image_path)
    run_inference_on_image(image_path)