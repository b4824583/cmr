import scipy.io
import torch
import numpy as np
anno_sfm = scipy.io.loadmat('misc/cachedir/cub/sfm/anno_testval.mat',struct_as_record=False, squeeze_me=True)
print(anno_sfm)
print("anno_sfm[S]3x15:",anno_sfm["S"][0])
print("len anno_sfm[S]3x15:",len(anno_sfm["S"][0]))


print("sfm_anno:",len(anno_sfm["sfm_anno"]))
print("conv_tri:",len(anno_sfm["conv_tri"]))
sfm_mean_shape = torch.Tensor(np.transpose(anno_sfm['S'])).cuda(device=0)

#print(sfm_mean_shape)



#anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)
#sfm_mean_shape = torch.Tensor(np.transpose(anno_sfm['S'])).cuda(device=opts.gpu_id)
#self.sfm_mean_shape = Variable(sfm_mean_shape, requires_grad=False)
#self.sfm_mean_shape = self.sfm_mean_shape.unsqueeze(0).repeat(opts.batch_size, 1, 1)
#sfm_face = torch.LongTensor(anno_sfm['conv_tri'] - 1).cuda(device=opts.gpu_id)
#self.sfm_face = Variable(sfm_face, requires_grad=False)
#faces = self.sfm_face.view(1, -1, 3)