import scipy.io
import torch
import numpy as np
import plotly.graph_objects as go
anno_sfm = scipy.io.loadmat('misc/cachedir/cub/sfm/anno_testval.mat',struct_as_record=False, squeeze_me=True)
#print(anno_sfm)
#print("anno_sfm[S]3x15:",anno_sfm["S"])
print("len anno_sfm[S]3x15:",len(anno_sfm["S"]))


#------------------------this is initial mean shape-----------------------

sfm_mean_shape = torch.Tensor(np.transpose(anno_sfm['S'])).cuda(device=0)


#print(sfm_mean_shape)
#----------------------------vertex
mesh_x=np.empty(len(anno_sfm["S"][0]))
mesh_y=np.empty(len(anno_sfm["S"][0]))
mesh_z=np.empty(len(anno_sfm["S"][0]))


#----------------------face-----------------
tri_i=np.empty(len(anno_sfm["conv_tri"]))
tri_j=np.empty(len(anno_sfm["conv_tri"]))
tri_k=np.empty(len(anno_sfm["conv_tri"]))

for i in range(len(anno_sfm["S"][0])):
	for j in range(3):
		if(j==0):			
			mesh_x[i]=anno_sfm["S"][j][i]
		elif(j==1):
			mesh_y[i]=anno_sfm["S"][j][i]
		else:
			mesh_z[i]=anno_sfm["S"][j][i]
print(len(mesh_z))
for i in range(len(anno_sfm["conv_tri"])):
	for j in range(3):
		if(j==0):
			tri_i[i]=anno_sfm["conv_tri"][i][j]-1
		elif(j==1):
			tri_j[i]=anno_sfm["conv_tri"][i][j]-1
		else:
			tri_k[i]=anno_sfm["conv_tri"][i][j]-1
			

fig = go.Figure(data=[go.Mesh3d(x=mesh_x, y=mesh_y, z=mesh_z, color='lightpink', opacity=0.5,i=tri_i,j=tri_j,k=tri_k)])
fig.show()

