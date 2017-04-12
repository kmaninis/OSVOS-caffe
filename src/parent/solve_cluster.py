from __future__ import division
import numpy as np
import sys
import os

caffe_root = '../../caffe-osvos/'
gpu_id = 0

sys.path.insert(0, caffe_root + 'python')
import caffe
print 'Caffe imported!!'

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        m, k, h, w = net.params[l][0].data.shape
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt


print "gpu_id is: "+str(gpu_id)
caffe.set_device(gpu_id)
caffe.set_mode_gpu()

model_name = '../../models/5stage-vgg.caffemodel'
iters = (15000, 15000, 20000)
for i in range(0,len(iters)):
	solver_name = './solvers/solver_step'+str(i+1)+'.prototxt'
	solver = caffe.SGDSolver(solver_name)
	#solver.restore('osvos_parent_step1_iter_10000.solverstate')
	solver.net.copy_from(model_name)
	interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
	interp_surgery(solver.net, interp_layers)
	solver.step(iters[i])
	model_name = './osvos_parent_step'+str(i+1)+'_iter_'+str(iters[i])+'.caffemodel'
