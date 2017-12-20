import tensorflow as tf
from model import Model
import cv2

graph_path = 'pero-model.meta'
checkpoints_path = './'

with tf.Session() as sess:
    nn = Model()
    nn.init(graph_path, checkpoints_path, sess)
    for i in range(1,7):
        image = cv2.imread('./data/test/bad/' + str(i) + '.jpg')
        print(nn.predict(image))
