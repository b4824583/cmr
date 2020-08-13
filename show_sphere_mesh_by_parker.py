from utils import mesh



import plotly.graph_objects as go
import numpy as np

# Download data set from plotly repo
pts = np.loadtxt(np.DataSource().open('https://raw.githubusercontent.com/plotly/datasets/master/mesh_dataset.txt'))
x, y, z = pts.T
print(type(x))
#fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, color='lightpink', opacity=0.50)])


#exit()
#------------------------test
verts, faces = mesh.create_sphere(3)


mesh_x=np.empty(len(verts))
mesh_y=np.empty(len(verts))
mesh_z=np.empty(len(verts))

tri_i=np.empty(len(faces))
tri_j=np.empty(len(faces))
tri_k=np.empty(len(faces))
print("verts:",verts[0])
print(faces)
for i in range(len(faces)):
	for j in range(3):
		if(j==0):
			tri_i[i]=faces[i][j]
		elif(j==1):
			tri_j[i]=faces[i][j]
		else:
			tri_k[i]=faces[i][j]

for i in range(len(verts)):
	for j in range(3):
		if(j==0):			
			mesh_x[i]=verts[i][j]
		elif(j==1):
			mesh_y[i]=verts[i][j]
		else:
			mesh_z[i]=verts[i][j]

fig = go.Figure(data=[go.Mesh3d(x=mesh_x, y=mesh_y, z=mesh_z, color='lightpink', opacity=0.5,i=tri_i,j=tri_j,k=tri_k)])
fig.show()

