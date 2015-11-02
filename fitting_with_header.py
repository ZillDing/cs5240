import numpy as np 
import numpy.linalg as la
file = open("example-2-without-noise_Right1.ply")   # the ply file without header
data = np.genfromtxt(file, delimiter=" ",skip_header=14)
file.close()
print data.shape

def header(num_vertices,dim):
    headers =""
    #dim == 9 => assume default data ... xyz (n)xyz rgb included in header 
    #dim == 7 => assume VCGLIB gen ply ... xyz rgb+alpha included in header
    if( (dim == 7) || (dim == 9)):
        headers ="ply\n"
        headers +="format ascii 1.0\n"
        if(dim == 7):
            headers +="comment VCGLIB generated\n"
        headers +="element vertex "+str(num_vertices)+"\n"
        headers +="property float x\n"
        headers +="property float y\n"
        headers +="property float z\n"
        if(dim == 9):
            headers+="property float nx"
            headers+="property float ny"
            headers+="property float nz"
        headers +="property uchar diffuse_red\n"
        headers +="property uchar diffuse_green\n"
        headers +="property uchar diffuse_blue\n"
        if(dim == 7):
            headers +="property uchar alpha\n"
        headers +="element face 0\n"
        headers +="property list uchar int vertex_indices\n"
        headers +="end_header\n"
    return headers

items=(0,3,6,10,15,21,28) # number of items in fitting with respecto the order of fitting

# Order =1: linear fit
# Order =2: quad fit
# Order =3: third order fit
# Order =4: fourth order fit
# Order =5: fifth order fit
# Order =6: sixth order fit
# Support orders up to 6
Order=1   
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

# Solve for least square solution
a,e,r,s = la.lstsq(D, z)
print "a =\n", a

# Compute fitting error
norm = la.norm(D * a - z)
err = norm * norm
print "Done fitting with order %d" % Order
print "err =", err
print "lstsq e =", e




#create mesh for interpolation 
x_min=np.min(data[:,0])
x_max=np.max(data[:,0])
y_min=np.min(data[:,1])
y_max=np.max(data[:,1])

num=500;  
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

#fill the color and other information with the value of its neareast neight
if True:
	print "Interpolating..."
	for k in xrange(num**2):
		idx=np.argmin(la.norm(data[:,0:3]-result[k,0:3],axis=1))
		result[k,3:]=data[idx,3:]
		if np.mod(k,10000)==0:
		    print "Processing "+str(k)
	print "Processed %d points." % (k+1)



print "Writing to result_Order_%d.ply" % Order
name = file.name
file = open("result_Order_%d_%s" % (Order,name), "w")   # write the data to file. To get a ply file, one needs to copy and paste the header and modify element vertex number by adding num^2

file.write(header(result.shape[0],dim))

if dim==10:
	np.savetxt(file, result,  ["%f", "%f", "%f","%f","%f","%f","%d","%d","%d","%d"], " ")
elif dim==9:
	np.savetxt(file, result,  ["%f", "%f", "%f","%f","%f","%f","%d","%d","%d"], " ")
elif dim==7:
	np.savetxt(file, result,  ["%f", "%f", "%f","%d","%d","%d","%d"], " ")
else:
	print "Please check the dimension of input data"
file.close()
