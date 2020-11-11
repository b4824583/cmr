"""
Mesh net model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

from utils import mesh
from utils import geometry as geom_utils
from . import net_blocks as nb
#import math

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_boolean('symmetric', True, 'Use symmetric mesh or not')
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')

flags.DEFINE_boolean('texture', True, 'if true uses texture!')
flags.DEFINE_boolean('symmetric_texture', True, 'if true texture is symmetric!')
flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face')

flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')

flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')

flags.DEFINE_boolean('only_mean_sym', False, 'If true, only the meanshape is symmetric')


#------------- Modules ------------#
#----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(n_blocks=4)
        self.enc_conv1 = nb.conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)

        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv.forward(img)

        out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc.forward(out_enc_conv1)

        return feat


class TexturePredictorUV(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, nz_feat, uv_sampler, opts, img_H=64, img_W=128, n_upconv=5, nc_init=256, predict_flow=False, symmetric=False, num_sym_faces=624):
        super(TexturePredictorUV, self).__init__()
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.symmetric = symmetric
        self.num_sym_faces = num_sym_faces
        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)
        self.predict_flow = predict_flow
        # B x F x T x T x 2 --> B x F x T*T x 2
        self.uv_sampler = uv_sampler.view(-1, self.F, self.T*self.T, 2)

        self.enc = nb.fc_stack(nz_feat, self.nc_init*self.feat_H*self.feat_W, 2)
        if predict_flow:
            nc_final=2
        else:
            nc_final=3
        self.decoder = nb.decoder2d(n_upconv, None, nc_init, init_fc=False, nc_final=nc_final, use_deconv=opts.use_deconv, upconv_mode=opts.upconv_mode)

    def forward(self, feat):
        # pdb.set_trace()
        uvimage_pred = self.enc.forward(feat)#net_bloks.fc_stack().forward
        uvimage_pred = uvimage_pred.view(uvimage_pred.size(0), self.nc_init, self.feat_H, self.feat_W)
        # B x 2 or 3 x H x W
        self.uvimage_pred = self.decoder.forward(uvimage_pred) #nb.decoder2d.forward 不確定是不是會跑到container的forward
        self.uvimage_pred = torch.tanh(self.uvimage_pred)

        tex_pred = torch.nn.functional.grid_sample(self.uvimage_pred, self.uv_sampler, align_corners=True)
        tex_pred = tex_pred.view(uvimage_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)

        if self.symmetric:
            #會走這一段，因為是對稱是true，回傳後會得到1 1280 6 6 2
            # Symmetrize.
            tex_left = tex_pred[:, -self.num_sym_faces:]
            return torch.cat([tex_pred, tex_left], 1)
        else:
            # Contiguous Needed after the permute..
            return tex_pred.contiguous()



class ShapePredictor(nn.Module):
    """
    Outputs mesh deformations
    """

    def __init__(self, nz_feat, num_verts):
        super(ShapePredictor, self).__init__()
        # self.pred_layer = nb.fc(True, nz_feat, num_verts)
        self.pred_layer = nn.Linear(nz_feat, num_verts * 3)

        # Initialize pred_layer weights to be small so initial def aren't so big
        self.pred_layer.weight.data.normal_(0, 0.0001)

    def forward(self, feat):
        # pdb.set_trace()
        delta_v = self.pred_layer.forward(feat)
        # Make it B x num_verts x 3
        delta_v = delta_v.view(delta_v.size(0), -1, 3)
        # print('shape: ( Mean = {}, Var = {} )'.format(delta_v.mean().data[0], delta_v.var().data[0]))
        return delta_v


class QuatPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot=4, classify_rot=False):
        super(QuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)
        self.classify_rot = classify_rot

    def forward(self, feat):
        quat = self.pred_layer.forward(feat)
        if self.classify_rot:
            quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)
        return quat


class ScalePredictor(nn.Module):
    def __init__(self, nz):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 1)

    def forward(self, feat):
        scale = self.pred_layer.forward(feat) + 1  #biasing the scale to 1
        scale = torch.nn.functional.relu(scale) + 1e-12
        # print('scale: ( Mean = {}, Var = {} )'.format(scale.mean().data[0], scale.var().data[0]))
        return scale


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, orth=True):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer.forward(feat)
        # print('trans: ( Mean = {}, Var = {} )'.format(trans.mean().data[0], trans.var().data[0]))
        return trans


class CodePredictor(nn.Module):
    def __init__(self, nz_feat=100, num_verts=1000):
        super(CodePredictor, self).__init__()
        self.quat_predictor = QuatPredictor(nz_feat)
        self.shape_predictor = ShapePredictor(nz_feat, num_verts=num_verts)
        self.scale_predictor = ScalePredictor(nz_feat)
        self.trans_predictor = TransPredictor(nz_feat)

    def forward(self, feat):
        shape_pred = self.shape_predictor.forward(feat)
        scale_pred = self.scale_predictor.forward(feat)
        quat_pred = self.quat_predictor.forward(feat)
        trans_pred = self.trans_predictor.forward(feat)
        return shape_pred, scale_pred, trans_pred, quat_pred

#------------ Mesh Net ------------#
#----------------------------------#

#----------------------- read off_file edited by parker
def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    for i_vert in range(n_verts):
        if(i_vert==0):
            str=[float(s) for s in file.readline().strip().split(' ')]
            verts = np.array([str])
        else:
            str = [float(s) for s in file.readline().strip().split(' ')]
            temp=np.array([str])
            verts=np.concatenate((verts,temp),axis=0)
    for i_face in range(n_faces):
        if(i_face==0):
            str=[int(s) for s in file.readline().strip().split(' ')]
            str.pop(0)
            faces = np.array([str])
        else:
            str = [int(s) for s in file.readline().strip().split(' ')]
            str.pop(0)
            temp=np.array([str])
            faces=np.concatenate((faces,temp),axis=0)
    return verts, faces
#--------------------
class MeshNet(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=100, num_kps=15, sfm_mean_shape=None):
        # Input shape is H x W of the image.
        #input shape is Turple (256,256)
        #這邊呼叫了mesh net的父類別，module.py 的Module.init()
        #在module創見模型，使用
        super(MeshNet, self).__init__()
        self.opts = opts
        self.pred_texture = opts.texture
        self.symmetric = opts.symmetric
        self.symmetric_texture = opts.symmetric_texture

        # Mean shape.
        #verts, faces = mesh.create_sphere(opts.subdivide)
        #---------------------------------------edited by parker
        #------------------------------------這邊是使用我們以建立的sphere mesh，不過可能實際上沒什麼用。

        f=open("sphere_mesh_use_mesh_lab.off")
        verts,faces=read_off(f)
        num_verts = verts.shape[0]

        # ------------------------------this is sphere mean shape----
        f=open("sphere_mesh.off","w")
        f.write("OFF\n")
        line=str(len(verts))+" "+str(len(faces))+" 0\n"
        f.write(line)
        mesh_x = np.empty(len(verts))
        mesh_y = np.empty(len(verts))
        mesh_z = np.empty(len(verts))
        for i in range(len(verts)):
            line=str(verts[i][0])+" "+str(verts[i][1])+" "+str(verts[i][2])+"\n"
            f.write(line)
            for j in range(3):
                if (j == 0):
                    mesh_x[i] = verts[i][j]
                elif (j == 1):
                    mesh_y[i] = verts[i][j]
                else:
                    mesh_z[i] = verts[i][j]
        tri_i = np.empty(len(faces))
        tri_j = np.empty(len(faces))
        tri_k = np.empty(len(faces))

        for i in range(len(faces)):
            face_point1=int(faces[i][0])
            face_point2=int(faces[i][1])
            face_point3=int(faces[i][2])

            line = str(3) + " " + str(face_point1) + " " + str(face_point2) + " " + str(face_point3) + "\n"


            # -------------------------------------------
            f.write(line)
            for j in range(3):
                if (j == 0):
                    tri_i[i] = faces[i][j]
                elif (j == 1):
                    tri_j[i] = faces[i][j]
                else:
                    tri_k[i] = faces[i][j]
        import plotly.graph_objects as go
#        fig = go.Figure(
#            data=[go.Mesh3d(x=mesh_x, y=mesh_y, z=mesh_z, color='lightpink', opacity=0.5, i=tri_i, j=tri_j, k=tri_k)])
#        fig.show()
        f.close()
        #--------------it is sphere mesh
        # ----------------------edited by parker-----------------

        if self.symmetric:
            verts, faces, num_indept, num_sym, num_indept_faces, num_sym_faces = mesh.make_symmetric(verts, faces)
            if sfm_mean_shape is not None:
                verts = geom_utils.project_verts_on_mesh(verts, sfm_mean_shape[0], sfm_mean_shape[1])

            #輸出number symmertic outpu是number symmetric以及 number  independent
            num_sym_output = num_indept + num_sym
            if opts.only_mean_sym:
                print('Only the mean shape is symmetric!')
                self.num_output = num_verts
            else:
                self.num_output = num_sym_output
            self.num_sym = num_sym
            self.num_indept = num_indept
            self.num_indept_faces = num_indept_faces
            self.num_sym_faces = num_sym_faces
            # mean shape is only half.
            #--------------------這邊很特別，因為以下是要輸出的vertex數量，它只輸出了一半的數量，目的是為了讓它可以對稱
            #shape is 337x3 是32個indenpent+305 symmetric vertices
            self.mean_v = nn.Parameter(torch.Tensor(verts[:num_sym_output]))
            # Needed for symmetrizing..
            #這邊做出一個[-1.,1.,1.],device="cuda:0"的矩陣
            self.flip = Variable(torch.ones(1, 3).cuda(), requires_grad=False)
            self.flip[0, 0] = -1
        else:
            if sfm_mean_shape is not None:
                verts = geom_utils.project_verts_on_mesh(verts, sfm_mean_shape[0], sfm_mean_shape[1])            
            self.mean_v = nn.Parameter(torch.Tensor(verts))
            self.num_output = num_verts
#---------------------------edited by parker
        # print("mesh_net verts:",verts)
        # print("mesh_net len:",len(verts))
#--------------------------------
        verts_np = verts
        faces_np = faces
        self.faces = Variable(torch.LongTensor(faces).cuda(), requires_grad=False)
        self.edges2verts = mesh.compute_edges2verts(verts, faces)

        vert2kp_init = torch.Tensor(np.ones((num_kps, num_verts)) / float(num_verts))
        # Remember initial vert2kp (after softmax)
        self.vert2kp_init = torch.nn.functional.softmax(Variable(vert2kp_init.cuda(), requires_grad=False), dim=1)
        self.vert2kp = nn.Parameter(vert2kp_init)


        self.encoder = Encoder(input_shape, n_blocks=4, nz_feat=nz_feat)
        self.code_predictor = CodePredictor(nz_feat=nz_feat, num_verts=self.num_output)

        if self.pred_texture:
            if self.symmetric_texture:
                num_faces = self.num_indept_faces + self.num_sym_faces
            else:
                num_faces = faces.shape[0]

            uv_sampler = mesh.compute_uvsampler(verts_np, faces_np[:num_faces], tex_size=opts.tex_size)
            # F' x T x T x 2
            uv_sampler = Variable(torch.FloatTensor(uv_sampler).cuda(), requires_grad=False)
            # B x F' x T x T x 2
            uv_sampler = uv_sampler.unsqueeze(0).repeat(self.opts.batch_size, 1, 1, 1, 1)
            img_H = int(2**np.floor(np.log2(np.sqrt(num_faces) * opts.tex_size)))
            img_W = 2 * img_H
            self.texture_predictor = TexturePredictorUV(
              nz_feat, uv_sampler, opts, img_H=img_H, img_W=img_W, predict_flow=True, symmetric=opts.symmetric_texture, num_sym_faces=self.num_sym_faces)
            nb.net_init(self.texture_predictor)

    def forward(self, img):
        img_feat = self.encoder.forward(img)# image_feat是1 200的shape
        codes_pred = self.code_predictor.forward(img_feat)#code preds don't know what's that
        if self.pred_texture:
            texture_pred = self.texture_predictor.forward(img_feat) #TexturePredictorUV.forward(image_feat)
            return codes_pred, texture_pred
        else:
            return codes_pred

    def symmetrize(self, V):
        """
        Takes num_indept+num_sym verts and makes it
        num_indept + num_sym + num_sym
        Is identity if model is not symmetric
        """
        if self.symmetric:
            if V.dim() == 2:
                # No batch
                V_left = self.flip * V[-self.num_sym:]
                return torch.cat([V, V_left], 0)
            else:
                # With batch
                V_left = self.flip * V[:, -self.num_sym:]
                return torch.cat([V, V_left], 1)
        else:
            return V

    def get_mean_shape(self):
        return self.symmetrize(self.mean_v)
