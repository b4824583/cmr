"""
Takes an image, returns stuff.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import scipy.misc
import torch
import torchvision
from torch.autograd import Variable
import scipy.io as sio

from nnutils import mesh_net
from nnutils import geom_utils
from nnutils.nmr import NeuralRenderer
from utils import bird_vis
import plotly.graph_objects as go
import math

# These options are off by default, but used for some ablations reported.
flags.DEFINE_boolean('ignore_pred_delta_v', False, 'Use only mean shape for prediction')
flags.DEFINE_boolean('use_sfm_ms', False, 'Uses sfm mean shape for prediction')
flags.DEFINE_boolean('use_sfm_camera', False, 'Uses sfm mean camera')


class MeshPredictor(object):
    def __init__(self, opts):
        self.opts = opts

        self.symmetric = opts.symmetric
        #img_size是(256,256)
        img_size = (opts.img_size, opts.img_size)
        print('Setting up model..')
        #-----------------目前猜測是在這一行的什後從mean mesh變成learned mesh的
#        print(opts.nz_feat)
#        exit()
        #nz_feat目前不確定是哪冒出來的，還要找源頭
        #nz_feat 為200
        self.model = mesh_net.MeshNet(img_size, opts, nz_feat=opts.nz_feat)
        #-----------------------------------經這一個之後就被改變了得到一個337的verts,但原本的verts至少有600個所以它可能是將某些點更動了,
        # 也可能是它會透過對稱的手法來變成完整的mean shape
        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        #model 從training()模式轉換成評估模式
        self.model.eval()

        self.model = self.model.cuda(device=self.opts.gpu_id)

        self.renderer = NeuralRenderer(opts.img_size)

        if opts.texture:#--------------------這個只是true而已
            self.tex_renderer = NeuralRenderer(opts.img_size)
            # Only use ambient light for tex renderer
            self.tex_renderer.ambient_light_only()
#--------------------------------這邊將initial mean shape拿進去訓練得到 訓練過後的learned mean shape
        #----------------是否使用use_sfm_ms(它門預設都沒有，這個mesh非常的簡陋，它必須經過學習才會得到一個mean shape
        if opts.use_sfm_ms:
            anno_sfm_path = osp.join(opts.cub_cache_dir, 'sfm', 'anno_testval.mat')
            anno_sfm = sio.loadmat(
                anno_sfm_path, struct_as_record=False, squeeze_me=True)
            sfm_mean_shape = torch.Tensor(np.transpose(anno_sfm['S'])).cuda(
                device=opts.gpu_id)
            self.sfm_mean_shape = Variable(sfm_mean_shape, requires_grad=False)
            self.sfm_mean_shape = self.sfm_mean_shape.unsqueeze(0).repeat(
                opts.batch_size, 1, 1)
            sfm_face = torch.LongTensor(anno_sfm['conv_tri'] - 1).cuda(
                device=opts.gpu_id)
            self.sfm_face = Variable(sfm_face, requires_grad=False)
            faces = self.sfm_face.view(1, -1, 3)
#-------------------------------------------
        else:
            # For visualization
            faces = self.model.faces.view(1, -1, 3)

        self.faces = faces.repeat(opts.batch_size, 1, 1)
        #--------------------------------------這邊會到vis render init()
        self.vis_rend = bird_vis.VisRenderer(opts.img_size,
                                             faces.data.cpu().numpy())
        self.vis_rend.set_bgcolor([1., 1., 1.])
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def load_network(self, network, network_label, epoch_label):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        network_dir = os.path.join(self.opts.checkpoint_dir, self.opts.name)
        save_path = os.path.join(network_dir, save_filename)
        print('loading {}..'.format(save_path))
        network.load_state_dict(torch.load(save_path))

        return

    def set_input(self, batch):
        opts = self.opts

        # original image where texture is sampled from.
        img_tensor = batch['img'].clone().type(torch.FloatTensor)

        # input_img is the input to resnet
        input_img_tensor = batch['img'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        self.input_imgs = Variable(
            input_img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.imgs = Variable(
            img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        if opts.use_sfm_camera:
            cam_tensor = batch['sfm_pose'].type(torch.FloatTensor)
            self.sfm_cams = Variable(
                cam_tensor.cuda(device=opts.gpu_id), requires_grad=False)

    def predict(self, batch):
        """
        batch has B x C x H x W numpy
        """
        self.set_input(batch)
        self.forward()
        return self.collect_outputs()

    def forward(self):
        if self.opts.texture:
            pred_codes, self.textures = self.model.forward(self.input_imgs)#這邊得到的textures就是1 1280 6 6 2
        else:
            pred_codes = self.model.forward(self.input_imgs)

        self.delta_v, scale, trans, quat = pred_codes

        if self.opts.use_sfm_camera:
            self.cam_pred = self.sfm_cams
        else:
            self.cam_pred = torch.cat([scale, trans, quat], 1)

        del_v = self.model.symmetrize(self.delta_v)
        # Deform mean shape:
        self.mean_shape = self.model.get_mean_shape()
#-------------------------edited by parker
#----------------------------這確實是mean shape----------------
        f=open("bird_mean_mesh.off","w")
        f.write("OFF\n")
        line=str(len(self.mean_shape))+" "+str(len(self.faces[0]))+" 0\n"
        f.write(line)
        mesh_x = np.empty(len(self.mean_shape))
        mesh_y = np.empty(len(self.mean_shape))
        mesh_z = np.empty(len(self.mean_shape))
        # print("bird_vis verts:", self.mean_shape)
        for i in range(len(self.mean_shape)):
            mesh_x_point=float(self.mean_shape[i][0])
            mesh_y_point=float(self.mean_shape[i][1])
            mesh_z_point=float(self.mean_shape[i][2])

            line=str(mesh_x_point)+" "+str(mesh_y_point)+" "+str(mesh_z_point)+"\n"
            f.write(line)
            for j in range(3):
                if (j == 0):
                    mesh_x[i] = self.mean_shape[i][j]
                elif (j == 1):
                    mesh_y[i] = self.mean_shape[i][j]
                else:
                    mesh_z[i] = self.mean_shape[i][j]

        tri_i = np.empty(len(self.faces[0]))
        tri_j = np.empty(len(self.faces[0]))
        tri_k = np.empty(len(self.faces[0]))

        for i in range(len(self.faces[0])):

            #-------------------------
            face_point1 = int(self.faces[0][i][0])
            face_point2 = int(self.faces[0][i][1])
            face_point3 = int(self.faces[0][i][2])
#--------------------------------------

            line = str(3) + " " + str(face_point1) + " " + str(face_point2) + " " + str(face_point3) + "\n"
            f.write(line)
            for j in range(3):
                if (j == 0):
                    tri_i[i] = self.faces[0][i][j]
                elif (j == 1):
                    tri_j[i] = self.faces[0][i][j]
                else:
                    tri_k[i] = self.faces[0][i][j]
#--------------我暫時不需要顯示這些東西
#        fig = go.Figure(
#            data=[go.Mesh3d(x=mesh_x, y=mesh_y, z=mesh_z, color='lightgreen', opacity=0.5,i=tri_i, j=tri_j, k=tri_k)])
#        fig.show()
        f.close()
#---------------------------------------------------------
#        exit()
        if self.opts.use_sfm_ms:
            self.pred_v = self.sfm_mean_shape
        elif self.opts.ignore_pred_delta_v:
            self.pred_v = self.mean_shape + del_v*0
        else:
            self.pred_v = self.mean_shape + del_v

        # Compute keypoints.
        if self.opts.use_sfm_ms:
            self.kp_verts = self.pred_v
        else:
            self.vert2kp = torch.nn.functional.softmax(
                self.model.vert2kp, dim=1)
            self.kp_verts = torch.matmul(self.vert2kp, self.pred_v)

        # Project keypoints
        self.kp_pred = self.renderer.project_points(self.kp_verts,
                                                    self.cam_pred)
        self.mask_pred = self.renderer.forward(self.pred_v, self.faces,
                                               self.cam_pred)

        # Render texture.
        if self.opts.texture and not self.opts.use_sfm_ms:
            if self.textures.size(-1) == 2:
                # Flow texture!
                self.texture_flow = self.textures
#-----------------------
                # txt_file = open("texture_flow.txt", "w")
                # txt_file.write(repr(self.textures.shape))
                # txt_file.write(repr(self.textures))
                # txt_file.close()
#-----------------------
                self.textures = geom_utils.sample_textures(self.textures,
                                                           self.imgs)
#-----------------------edited by parker
                # txt_file=open("texture_sample_textures.txt","w")
                # txt_file.write(repr(self.textures.shape))
                # txt_file.write(repr(self.textures))
                # txt_file.close()

            if self.textures.dim() == 5:  # B x F x T x T x 3
                tex_size = self.textures.size(2)
                self.textures = self.textures.unsqueeze(4).repeat(1, 1, 1, 1,
                                                                  tex_size, 1)#這一行部知道在幹麻

            # Render texture:
            self.texture_pred = self.tex_renderer.forward(
                self.pred_v, self.faces, self.cam_pred, textures=self.textures)

            # B x 2 x H x W
            uv_flows = self.model.texture_predictor.uvimage_pred
            # B x H x W x 2
            self.uv_flows = uv_flows.permute(0, 2, 3, 1)
            self.uv_images = torch.nn.functional.grid_sample(self.imgs,
                                self.uv_flows, align_corners=True)
            #edited_by parker
            # uv_flows=open("uv_flows.txt","w")
            # uv_flows.write(repr(self.uv_flows.shape))
            # uv_flows.write(repr(self.uv_flows))
            # uv_flows.close()
            # uv_images=open("uv_images.txt","w")
            # uv_images.write(repr(self.uv_images[0].shape))
            # uv_images_png=np.reshape(self.uv_images[0],(128,256,3))
            # uv_images.write(repr(uv_images_png))
            # uv_images.close()
            #---------------------
            #----------------------------------show uv image------ parker
            uv_image_array = np.zeros([128, 256, 3])

            for i in range(len(self.uv_images[0])):
                for j in range(len(self.uv_images[0][i])):
                    for k in range(len(self.uv_images[0][i][j])):
                        uv_image_array[j][k][i]=self.uv_images[0][i][j][k]
            import matplotlib.pyplot as plt
            plt.imshow(uv_image_array)
            plt.draw()
            plt.show()
            plt.savefig('uv_image_test.png')
            #----------------------------------
        else:
            self.textures = None

    def collect_outputs(self):
        outputs = {
            'kp_pred': self.kp_pred.data,
            'verts': self.pred_v.data,
            'kp_verts': self.kp_verts.data,
            'cam_pred': self.cam_pred.data,
            'mask_pred': self.mask_pred.data,
        }
        if self.opts.texture and not self.opts.use_sfm_ms:
            outputs['texture'] = self.textures
            outputs['texture_pred'] = self.texture_pred.data
            outputs['uv_image'] = self.uv_images.data
            outputs['uv_flow'] = self.uv_flows.data

        return outputs
