import re
import numpy as np
import numpy.linalg as la
import sys
import os,sys
import Image
import math


def get_header(file):
	count = 0
	header = ""
	for line in file:
		count += 1
		header += line
		if line.strip() == "end_header":
			break
	return (header, count)


if not len(sys.argv)==4:
	print 'Wrong number of arguments'
	print 'Usage: '+sys.argv[0]+' input_file order input_img'
	exit(1)


input_file= sys.argv[1]
Order= int(sys.argv[2])
output_file=input_file+".order-%d-color-fit.ply" % Order
img_add=sys.argv[3]
#P=[[26357,740,-838,-733], [1530,-26641,-174,-1683], [0,0,0,1]]
P=np.genfromtxt(input_file+'.pmatrix.txt', delimiter=' ')
print 'input_file:'+input_file
print 'output_file'+output_file




# order = 1: linear fit
# order = 2: quad fit
# order = 3: third order fit
# order = 4: fourth order fit
# order = 5: fifth order fit
# order = 6: sixth order fit
# Support orders up to 6
#order = 1
num = 500
#input_file = "data_dzy2/example-1-left-denoise.ply"
#output_file = "data_dzy2/example-1-left-result-order-%d.ply" % order


file = open(input_file)
header_tuple = get_header(file)
data = np.genfromtxt(file, delimiter=" ", skip_header=header_tuple[1])
file.close()
print data.shape

items=(0,3,6,10,15,21,28) # number of items in fitting with respecto the order of fitting

# Order =1: linear fit
# Order =2: quad fit
# Order =3: third order fit
# Order =4: fourth order fit
# Order =5: fifth order fit
# Order =6: sixth order fit
# Support orders up to 6

#Order=2
item=items[Order]

z = np.matrix(data[:,2]).T
n = len(data[:,0])
D = np.matrix(np.empty([n,item]))

D[:,0] = np.ones([n,1])
D[:,1:3] = data[:,0:2]
if Order>1:
	D[:,3:5] = data[:,0:2]**2
	D[:,5] = np.matrix(data[:,0]*data[:,1]).T
if Order>2:
	D[:,6:8] = data[:,0:2]**3;
	D[:,8] = np.matrix(data[:,0]**2 *data[:,1]).T;
	D[:,9] = np.matrix(data[:,0]* data[:,1]**2).T;
if Order>3:
	D[:,10:12]= data[:,0:2]**4;
	D[:,12] = np.matrix(data[:,0]**3 *data[:,1]).T;
	D[:,13] = np.matrix(data[:,0]**2 *data[:,1]**2).T;
	D[:,14] = np.matrix(data[:,0] * data[:,1]**3).T;
if Order>4:
	D[:,15:17]= data[:,0:2]**5;
	D[:,17] = np.matrix(data[:,0]**4 *data[:,1]).T;
	D[:,18] = np.matrix(data[:,0]**3 *data[:,1]**2).T;
	D[:,19] = np.matrix(data[:,0]**2 *data[:,1]**3).T;
	D[:,20] = np.matrix(data[:,0] *data[:,1]**4).T;
if Order>5:
	D[:,21:23]= data[:,0:2]**6;
	D[:,23] = np.matrix(data[:,0]**5 *data[:,1]).T;
	D[:,24] = np.matrix(data[:,0]**4 *data[:,1]**2).T;
	D[:,25] = np.matrix(data[:,0]**3 *data[:,1]**3).T;
	D[:,26] = np.matrix(data[:,0]**2 *data[:,1]**4).T;
	D[:,27] = np.matrix(data[:,0] *data[:,1]**5).T;




# Print to verify that data is arranged correctly.
print "D =\n", D
print "v =\n", z
D_hat=np.mat(np.copy(D))
z_hat=np.mat(np.copy(z))
# prepare the weighted dat
W=np.ones(n)
p=0.5

for iter in xrange(20):
	W=np.sqrt(W)
	for i in xrange(n):
		D_hat[i,:]=W[i]*D[i,:]
		z_hat[i]=W[i]*z[i]

	# Solve for least square solution
	a,e,r,s = la.lstsq(D_hat, z_hat)
	print "a =\n", a
	W=(np.ravel(abs(z-D*a)))**p
	# Compute fitting error
	norm = la.norm(D.dot(a) - z)
	err = norm * norm
	print "Iter %d of fitting with order %d" % (iter,Order)
	print "err =", err
	norm = la.norm(D_hat.dot(a) - z_hat)
	err = norm * norm
	print "err =", err
	print "lstsq e =", e



x_min=np.min(data[:,0])
x_max=np.max(data[:,0])
y_min=np.min(data[:,1])
y_max=np.max(data[:,1])

x_arr=np.linspace(x_min, x_max, num=num, endpoint=True)
y_arr=np.linspace(y_min, y_max, num=num, endpoint=True)
xv, yv = np.meshgrid(x_arr, y_arr)

data_new=np.concatenate([[xv.reshape(num**2,)], [yv.reshape(num**2,)]]).T
n = len(data_new[:,0])
D = np.matrix(np.empty([n,item]))

D[:,0] = np.ones([n,1])
D[:,1:3] = data_new[:,0:2]
if Order>1:
	D[:,3:5] = data_new[:,0:2]**2
	D[:,5] = np.matrix(data_new[:,0]*data_new[:,1]).T
if Order>2:
	D[:,6:8] = data_new[:,0:2]**3;
	D[:,8] = np.matrix(data_new[:,0]**2 *data_new[:,1]).T;
	D[:,9] = np.matrix(data_new[:,0]* data_new[:,1]**2).T;
if Order>3:
	D[:,10:12]= data_new[:,0:2]**4;
	D[:,12] = np.matrix(data_new[:,0]**3 *data_new[:,1]).T;
	D[:,13] = np.matrix(data_new[:,0]**2 *data_new[:,1]**2).T;
	D[:,14] = np.matrix(data_new[:,0] * data_new[:,1]**3).T;
if Order>4:
	D[:,15:17]= data_new[:,0:2]**5;
	D[:,17] = np.matrix(data_new[:,0]**4 *data_new[:,1]).T;
	D[:,18] = np.matrix(data_new[:,0]**3 *data_new[:,1]**2).T;
	D[:,19] = np.matrix(data_new[:,0]**2 *data_new[:,1]**3).T;
	D[:,20] = np.matrix(data_new[:,0] *data_new[:,1]**4).T;
if Order>5:
	D[:,21:23]= data_new[:,0:2]**6;
	D[:,23] = np.matrix(data_new[:,0]**5 *data_new[:,1]).T;
	D[:,24] = np.matrix(data_new[:,0]**4 *data_new[:,1]**2).T;
	D[:,25] = np.matrix(data_new[:,0]**3 *data_new[:,1]**3).T;
	D[:,26] = np.matrix(data_new[:,0]**2 *data_new[:,1]**4).T;
	D[:,27] = np.matrix(data_new[:,0] *data_new[:,1]**5).T;


pred=D*a

dim=data.shape[1];
result = np.matrix(np.empty([num**2,dim]))
result[:,2] = pred
result[:,0:2] = data_new


img= Image.open(img_add)
X=np.ones((4))
oor=0
if True:
	print "Interpolating..."
	for k in xrange(num**2):
		X[0:3]=result[k,0:3]
		x=X.dot(P[0,:])/X.dot(P[2,:])
		y=X.dot(P[1,:])/X.dot(P[2,:])
		x1=math.floor(x)
		x2=math.ceil(x)
		y1=math.floor(y)
		y2=math.ceil(y)
		idx=np.argmin(la.norm(data[:,0:3]-result[k,0:3],axis=1))
		result[k,3:]=data[idx,3:]
		if x2<img.size[0] and x1>-1 and y1> -1 and y2<img.size[1]:
			#print x,y,img.size[0],img.size[1]
			[R11,G11,B11]=img.getpixel((x1,y1))
			[R12,G12,B12]=img.getpixel((x1,y2))
			[R21,G21,B21]=img.getpixel((x2,y1))
			[R22,G22,B22]=img.getpixel((x2,y2))
			w11=(x-x1)*(y-y1)
			w12=(x-x1)*(y2-y)
			w21=(x2-x)*(y-y1)
			w22=(x2-x)*(y2-y)
			result[k,6]=round(R11*w11+R12*w12+R21*w21+R22*w22)
			result[k,7]=round(G11*w11+G12*w12+G21*w21+G22*w22)
			result[k,8]=round(B11*w11+B12*w12+B21*w21+B22*w22)
		else:
			oor=oor+1
		if np.mod(k,1000)==0:
			print "Processing %d over %d" %(k,num**2)			
			print "Out of range points: %d " % (oor)
	print "Processed %d points." % (k+1)
	print "Out of range points: %d " % (oor)



print "Writing to", output_file

# write the data to file
file = open(output_file, "w")

# compute output data format
data_format = ["%f", "%f", "%f", "%f", "%f", "%f"]
for i in range(dim - len(data_format)):
	data_format.append("%d")
# compute output ply file header
output_header = re.sub(r"(element vertex) [0-9]*",
	r"\1 " + str(num**2),
	header_tuple[0].strip())

output_header = re.sub(r"(element face) [0-9]*",
	r"\1 " + str((num-1)**2),
	output_header)

np.savetxt(file, result,
	fmt=data_format,
	delimiter=" ",
	header=output_header,
	comments="")



face_list=4*np.ones([(num-1)**2,5])
cc=0
for r in xrange(num-1):
	for c in xrange(num-1):
		face_list[cc,1]=r+num*c
		face_list[cc,2]=r+1+num*c
		face_list[cc,3]=r+1+num*(c+1)
		face_list[cc,4]=r+num*(c+1)
		cc=cc+1


data_format = ["%d", "%d", "%d", "%d","%d"]
np.savetxt(file, face_list,fmt=data_format,delimiter=" ")

file.close()
