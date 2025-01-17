"""
Demo of CMR.

Note that CMR assumes that the object has been detected, so please use a picture of a bird that is centered and well cropped.

Sample usage:

python demo.py --name bird_net --num_train_epoch 500 --img_path misc/demo_data/img1.jpg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
import numpy as np
import skimage.io as io

import torch

from nnutils import test_utils
from nnutils import predictor as pred_util
from utils import image as img_util
#----------------------------edited by parker call to graph
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
#
# with PyCallGraph(output=GraphvizOutput()):
#     code_to_profile()
#-------------------------


flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_integer('img_size', 256, 'image size the network was trained on.')

opts = flags.FLAGS


def preprocess_image(img_path, img_size=256):
    img = io.imread(img_path) / 255.
    # Scale the max image size to be img_size
    #-這邊將圖片的大小scale到257
    scale_factor = float(img_size) / np.max(img.shape[:2])
    img, _ = img_util.resize_img(img, scale_factor)#256x256x3

    # Crop img_size x img_size from the center
    #---------------其實看不太懂它為什麼要切割，因為它切割的大小是257x257，而它縮放的大小是256x256
    #--------------他是不是在耍人阿！？？
    #--------------切割是由中心點往外切出一個bounding box
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
#p
#    print("center1:"+str(center))
    # img center in (x, y)
    center = center[::-1]
    #p
#    print("center2:"+str(center))

    bbox = np.hstack([center - img_size / 2., center + img_size / 2.])
    #p
#    print("bbox:"+str(bbox))

    img = img_util.crop(img, bbox, bgval=1.)#257x257x3

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))#3x257x257

    return img


def visualize(img, outputs, renderer):
    vert = outputs['verts'][0]
    cam = outputs['cam_pred'][0]

    texture = outputs['texture'][0]
    # -----------------------這邊會輸出已經預測好的bird mesh
    # -----------------------------------------這邊會Vis Render call()
    shape_pred = renderer(vert, cam)
    # -----------------------這邊會輸出已經預測好的bird mesh
    # -----------------------------------------這邊會Vis Render call()

    img_pred = renderer(vert, cam, texture=texture)

    # Different viewpoints.
    vp1 = renderer.diff_vp(
        vert, cam, angle=30, axis=[0, 1, 0], texture=texture, extra_elev=True)
    vp2 = renderer.diff_vp(
        vert, cam, angle=60, axis=[0, 1, 0], texture=texture, extra_elev=True)
    vp3 = renderer.diff_vp(
        vert, cam, angle=60, axis=[1, 0, 0], texture=texture)
    # f=open("texture.txt","w")
    # f.write(repr(texture.shape)+"\n")
    # f.write(repr(texture))
    # f.close()
    img = np.transpose(img, (1, 2, 0))
    import matplotlib.pyplot as plt
    # plt.ion()
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(texture)
    # plt.show()
    # plt.savefig("texture.png")
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(shape_pred)
    plt.title('pred mesh')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(img_pred)
    plt.title('pred mesh w/texture')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(vp1)
    plt.title('different viewpoints')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(vp2)
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(vp3)
    plt.axis('off')
    plt.draw()
    plt.show()
    print('saving file to demo.png')
    plt.savefig('demo.png')


def main(_):
#----edited by parker  call to graph
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'call_to_graph.png'


    img = preprocess_image(opts.img_path, img_size=opts.img_size)
    print("opts:",opts.gpu_id)


#    with PyCallGraph(output=graphviz):
#img的維度是3x257x257
# 創建一個pytorch tensor 的batch 維度是["img"][1][3][257][257]而且值為 1.0

    batch = {'img': torch.Tensor(np.expand_dims(img, 0))}
    predictor = pred_util.MeshPredictor(opts)
    #-----------------得到預測好的vertice
    outputs = predictor.predict(batch)
    #-----------------------------draw predited mesh

    #----------------------------------

    # This is resolution
    renderer = predictor.vis_rend
    renderer.set_light_dir([0, 1, -1], 0.4)
    # output["verts"]是已經預測好的vertce
    visualize(img, outputs, predictor.vis_rend)


if __name__ == '__main__':
    opts.batch_size = 1
    opts.name="bird_net"
    opts.num_train_epoch=500
    opts.img_path="misc/demo_data/img2.jpg"
    app.run(main)
