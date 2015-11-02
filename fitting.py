import numpy as np
import numpy.linalg as la

def get_header(file):
	count = 0
	header = ""
	for line in file:
		count += 1
		header += line
		if line.strip() == "end_header":
			break
	return (header, count)

# order = 1: linear fit
# order = 2: quad fit
# order = 3: third order fit
# order = 4: fourth order fit
# order = 5: fifth order fit
# order = 6: sixth order fit
# Support orders up to 6
order = 3
input_file = "data/example-1-without-noise.ply"
output_file = "data/example-1-result-order-%d.ply" % order


file = open(input_file)
header_tuple = get_header(file)
data = np.genfromtxt(file, delimiter=" ", skip_header=header_tuple[1])
file.close()
print data.shape

items = (0, 3, 6, 10, 15, 21, 28) # number of items in fitting with respecto the order of fitting
item = items[order]

z = np.matrix(data[:,2]).T
n = len(data[:,0])
D = np.matrix(np.empty([n,item]))

D[:,0] = np.ones([n,1])
D[:,1:3] = data[:,0:2]
if order > 1:
	D[:,3:5] = data[:,0:2]**2
	D[:,5] = np.matrix(data[:,0] * data[:,1]).T
if order > 2:
	D[:,6:8] = data[:,0:2]**3;
	D[:,8] = np.matrix(data[:,0]**2 * data[:,1]).T;
	D[:,9] = np.matrix(data[:,0] * data[:,1]**2).T;
if order > 3:
	D[:,10:12] = data[:,0:2]**4;
	D[:,12] = np.matrix(data[:,0]**3 * data[:,1]).T;
	D[:,13] = np.matrix(data[:,0]**2 * data[:,1]**2).T;
	D[:,14] = np.matrix(data[:,0] * data[:,1]**3).T;
if order > 4:
	D[:,15:17] = data[:,0:2]**5;
	D[:,17] = np.matrix(data[:,0]**4 * data[:,1]).T;
	D[:,18] = np.matrix(data[:,0]**3 * data[:,1]**2).T;
	D[:,19] = np.matrix(data[:,0]**2 * data[:,1]**3).T;
	D[:,20] = np.matrix(data[:,0] * data[:,1]**4).T;
if order > 5:
	D[:,21:23] = data[:,0:2]**6;
	D[:,23] = np.matrix(data[:,0]**5 * data[:,1]).T;
	D[:,24] = np.matrix(data[:,0]**4 * data[:,1]**2).T;
	D[:,25] = np.matrix(data[:,0]**3 * data[:,1]**3).T;
	D[:,26] = np.matrix(data[:,0]**2 * data[:,1]**4).T;
	D[:,27] = np.matrix(data[:,0] * data[:,1]**5).T;

# Print to verify that data is arranged correctly.
print "D =\n", D
print "v =\n", z

# Solve for least square solution
a,e,r,s = la.lstsq(D, z)
print "a =\n", a

# Compute fitting error
norm = la.norm(D * a - z)
err = norm * norm
print "Done fitting with order:", order
print "err =", err
print "lstsq e =", e


#create mesh for interpolation
x_min = np.min(data[:,0])
x_max = np.max(data[:,0])
y_min = np.min(data[:,1])
y_max = np.max(data[:,1])

num = 500;
x_arr = np.linspace(x_min, x_max, num=num, endpoint=True)
y_arr = np.linspace(y_min, y_max, num=num, endpoint=True)
xv, yv = np.meshgrid(x_arr, y_arr)

data_new = np.concatenate([[xv.reshape(num**2,)], [yv.reshape(num**2,)]]).T
n = len(data_new[:,0])
D = np.matrix(np.empty([n,item]))

D[:,0] = np.ones([n,1])
D[:,1:3] = data_new[:,0:2]
if order > 1:
	D[:,3:5] = data_new[:,0:2]**2
	D[:,5] = np.matrix(data_new[:,0] * data_new[:,1]).T
if order > 2:
	D[:,6:8] = data_new[:,0:2]**3;
	D[:,8] = np.matrix(data_new[:,0]**2 * data_new[:,1]).T;
	D[:,9] = np.matrix(data_new[:,0] * data_new[:,1]**2).T;
if order > 3:
	D[:,10:12] = data_new[:,0:2]**4;
	D[:,12] = np.matrix(data_new[:,0]**3 * data_new[:,1]).T;
	D[:,13] = np.matrix(data_new[:,0]**2 * data_new[:,1]**2).T;
	D[:,14] = np.matrix(data_new[:,0] * data_new[:,1]**3).T;
if order > 4:
	D[:,15:17] = data_new[:,0:2]**5;
	D[:,17] = np.matrix(data_new[:,0]**4 * data_new[:,1]).T;
	D[:,18] = np.matrix(data_new[:,0]**3 * data_new[:,1]**2).T;
	D[:,19] = np.matrix(data_new[:,0]**2 * data_new[:,1]**3).T;
	D[:,20] = np.matrix(data_new[:,0] * data_new[:,1]**4).T;
if order > 5:
	D[:,21:23] = data_new[:,0:2]**6;
	D[:,23] = np.matrix(data_new[:,0]**5 * data_new[:,1]).T;
	D[:,24] = np.matrix(data_new[:,0]**4 * data_new[:,1]**2).T;
	D[:,25] = np.matrix(data_new[:,0]**3 * data_new[:,1]**3).T;
	D[:,26] = np.matrix(data_new[:,0]**2 * data_new[:,1]**4).T;
	D[:,27] = np.matrix(data_new[:,0] * data_new[:,1]**5).T;

pred = D * a

dim = data.shape[1];
result = np.matrix(np.empty([num**2, dim]))
result[:,2] = pred
result[:,0:2] = data_new

# fill the color and other information with the value of its neareast neight
if True:
	print "Interpolating..."
	for k in xrange(num**2):
		idx = np.argmin(la.norm(data[:,0:3] - result[k,0:3], axis=1))
		result[k,3:] = data[idx,3:]
		if np.mod(k, 5000) == 0:
			print "Processing", k
	print "Processed %d points." % (k + 1)

print "Writing to", output_file
# write the data to file. To get a ply file, one needs to copy and paste the header and modify element vertex number by adding num^2
file = open(output_file, "w")

data_format = None
if dim == 11:
	data_format = ["%f", "%f", "%f","%f","%f","%f","%d","%d","%d","%d","%d"]
if dim == 10:
	data_format = ["%f", "%f", "%f","%f","%f","%f","%d","%d","%d","%d"]
elif dim == 9:
	data_format = ["%f", "%f", "%f","%f","%f","%f","%d","%d","%d"]

if data_format is None:
	print "Please check the dimension of input data. Unsupported dimension:", dim
else:
	np.savetxt(file, result,  data_format, " ", header=header_tuple[0].strip(), comments="")

file.close()
