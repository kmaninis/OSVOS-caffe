from __future__ import division
import numpy as np
import sys
import os
#os.chdir('/srv/glusterfs/kmaninis/caffe_experiments/DAVIS-mask-separate-mixed-neg-norot-ext-finetune/')
caffe_root = '/home/eec/Documents/external/deep_learning/OSVOS-matcaffe/caffe_osvos/'
#caffe_root = '/home/kmaninis/scratch/hed_mod/' 
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

gpu_id = 0
print "gpu_id is: "+str(gpu_id)
caffe.set_device(gpu_id)

caffe.set_mode_gpu()

model_name = '/home/eec/Documents/external/deep_learning/OSVOS-matcaffe/models/OSVOS_parent.caffemodel'
solver_name = '/home/eec/Documents/external/deep_learning/OSVOS-matcaffe/models/solvers/solver_drift-chicane_python.prototxt'
solver = caffe.SGDSolver(solver_name)
#solver.restore('drive_iter_18000.solverstate')
solver.net.copy_from(model_name)
solver.step(2000)
