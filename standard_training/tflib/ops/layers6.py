import numpy as np
import tensorflow as tf
import tflib as lib
from tensorflow.python.framework import ops

my_ind = tf.load_op_library('cuda/indices_op.so')
my_gat = tf.load_op_library('cuda/gat_op.so')
my_gat1 = tf.load_op_library('cuda/gat1_op.so')

my_hist = tf.load_op_library('cuda/hist_op.so')
my_hist1 = tf.load_op_library('cuda/hist1_op.so')

@tf.RegisterGradient("Gat")
def _gat_grad(op, grad0):
    grad = my_gat1.gat1(grad0, op.inputs[0], op.inputs[1], op.inputs[2],  op.inputs[3],  op.get_attr("ndepth"))
    return [grad, None, None, None]


@tf.RegisterGradient("Hist")
def _hist_grad(op, grad0, grad1):
    grad = my_hist1.hist1(grad0, op.inputs[0], op.outputs[1], op.get_attr("nbins"), op.get_attr("nbatch"),
                          op.get_attr("ndepth"), op.get_attr("alpha"))
    return [grad]

ops.NotDifferentiable('Indices')
ops.NotDifferentiable('Gat1')


def conv2D(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def deconv2D(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def bn(x, epsilon=1e-5, momentum=0.9, name="batch_norm", train=True):
    with tf.variable_scope(name):
        out = tf.contrib.layers.batch_norm(x,
                                           decay=momentum,
                                           updates_collections=None,
                                           epsilon=epsilon,
                                           scale=True,
                                           is_training=train,
                                           scope=name)
    return out

def conv2d(x, filter_shape, bias=True, stride=1, padding="SAME", name="conv2d"):
    kw, kh, nin, nout = filter_shape
    pad_size = (kw - 1) / 2

    if padding == "VALID":
        x = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], "SYMMETRIC")

    initializer = tf.random_normal_initializer(0., 0.02)
    with tf.variable_scope(name):
        weight = tf.get_variable("weight", shape=filter_shape, initializer=initializer)
        x = tf.nn.conv2d(x, weight, [1, stride, stride, 1], padding=padding)

        if bias:
            b = tf.get_variable("bias", shape=filter_shape[-1], initializer=tf.constant_initializer(0.))
            x = tf.nn.bias_add(x, b)
    return x


def fc(x, output_shape, bias=True, name='fc'):
    shape = x.get_shape().as_list()
    dim = np.prod(shape[1:])
    x = tf.reshape(x, [-1, dim])
    input_shape = dim

    initializer = tf.random_normal_initializer(0., 0.02)
    with tf.variable_scope(name):
        weight = tf.get_variable("weight", shape=[input_shape, output_shape], initializer=initializer)
        x = tf.matmul(x, weight)

        if bias:
            b = tf.get_variable("bias", shape=[output_shape], initializer=tf.constant_initializer(0.))
            x = tf.nn.bias_add(x, b)
    return x


def Pool(x, r=2, s=1):
    return tf.nn.avg_pool(x, ksize=[1, r, r, 1], strides=[1, s, s, 1], padding="SAME")


def l1_loss(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def l2_loss(x, y):
    return tf.reduce_mean(tf.nn.l2_loss(x - y))

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def optimal(source, target, sbatch, tbatch, ndepth, nbins, alpha):
    # compute max min
    src_max = tf.reduce_max(source, reduction_indices=[0])
    src_min = tf.reduce_min(source, reduction_indices=[0])
    tar_max = tf.reduce_max(target, reduction_indices=[0])
    tar_min = tf.reduce_min(target, reduction_indices=[0])

    # compute histogram
    src_scal = (source - src_min) / (src_max - src_min) * nbins
    tar_scal = (target - tar_min) / (tar_max - tar_min) * nbins

    src_hist, _ = my_hist.hist(src_scal, nbins, sbatch, ndepth, alpha)
    tar_hist, _ = my_hist.hist(tar_scal, nbins, tbatch, ndepth, alpha)

    # compute cumsum
    src_cumsum = tf.cumsum(src_hist, 0) / sbatch
    tar_cumsum = tf.cumsum(tar_hist, 0) / tbatch

    # compute indices
    nn_idx1, idx11, idx12, nn_idx2, idx21, idx22 = my_ind.indices(src_scal, src_cumsum, tar_cumsum,
                                                                  sbatch, ndepth, nbins)
    # f = interp1(PY, 0:nbins, PX, 'linear'); (corresponding to matlab code from original author)
    tar_gat12 =  my_gat.gat(tar_cumsum, idx12, tf.constant(nbins + 1), tf.constant(nbins + 1),
                            ndepth)
    tar_gat11 =  my_gat.gat(tar_cumsum, idx11, tf.constant(nbins + 1), tf.constant(nbins + 1),
                            ndepth)
    f_prime = tar_gat12 - tar_gat11 + 1e-3
    f = 1.0 / f_prime * (src_cumsum - tar_gat11) + nn_idx1

    # g = interp1(u, f', D0R(i,:)); (corresponding to matlab code from original author)
    f_gat12 =  my_gat.gat(f, idx22, tf.constant(nbins + 1), tf.constant(sbatch),  ndepth)
    f_gat11 =  my_gat.gat(f, idx21, tf.constant(nbins + 1), tf.constant(sbatch),  ndepth)
    g_prime = f_gat12 - f_gat11
    g = g_prime * (src_scal - nn_idx2) + f_gat11 - 1

    src_new = g * (tar_max - tar_min) / nbins + tar_min
    return src_new

def SWDBlock(name, source, target, nbins=32, alpha=-1):
    src_shape = source.get_shape().as_list()
    tar_shape = target.get_shape().as_list()
    sbatch = src_shape[0]
    tbatch = tar_shape[0]
    ndepth = np.prod(src_shape[1:])

    with tf.name_scope(name) as scope:
        weight_value, _ = tf.qr(tf.random_normal([ndepth, ndepth]))
        weight = lib.param(name+'.swd_proj', weight_value)

        src_proj = tf.matmul(tf.reshape(source, [-1, ndepth]), weight)
        tar_proj = tf.matmul(tf.reshape(target, [-1, ndepth]), weight)

        src_new = optimal(src_proj, tar_proj, sbatch, tbatch, ndepth, nbins, alpha)
        src_out = tf.matmul(src_new, tf.transpose(weight))
    return src_out

