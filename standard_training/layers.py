import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops

my_ind = tf.load_op_library('./cuda/indices_op.so')
my_gat = tf.load_op_library('./cuda/gat_op.so')
my_gat1 = tf.load_op_library('./cuda/gat1_op.so')

my_hist = tf.load_op_library('./cuda/hist_op.so')
my_hist1 = tf.load_op_library('./cuda/hist1_op.so')

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
ops.NotDifferentiable('Hist1')

def conv2D(input_, output_dim,
           k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
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

def l1_gradient(x, y):
    x_sh = x.get_shape().as_list()
    x = tf.reshape(x, [x_sh[0], 32, 32, 3])
    y = tf.reshape(y, [x_sh[0], 32, 32, 3])
    dx = tf.constant([[0., 0., 0.,], [1., 0., -1], [0., 0., 0.]])
    w_dx = tf.tile(tf.reshape(dx, [3,3,1,1]), [1, 1, 3, 1])
    w_dy = tf.tile(tf.reshape(tf.transpose(dx), [3,3,1,1]), [1, 1, 3, 1])
    w_dxy = tf.concat([w_dx, w_dy], 3)
    grad_x = tf.nn.conv2d(x, w_dxy, [1, 1, 1, 1], padding='VALID')
    grad_y = tf.nn.conv2d(y, w_dxy, [1, 1, 1, 1], padding='VALID')
    return tf.reduce_mean(tf.abs(grad_x - grad_y))

def l2_loss(x, y):
    return tf.reduce_mean(tf.square(x - y))

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

def project_l1(source, target, fz=3, nbins=32, alpha=-1, name='idt'):
    f_num = fz * fz * 3


    wei, _ = tf.qr(tf.random_normal([f_num, f_num]))
    weight = tf.reshape(wei, [fz,fz,3,f_num])
    src_proj = tf.nn.conv2d(source, weight, strides=[1, fz, fz, 1], padding='VALID')
    tar_proj = tf.nn.conv2d(target, weight, strides=[1, fz, fz, 1], padding='VALID')

    src_shape = src_proj.get_shape().as_list()
    tar_shape = tar_proj.get_shape().as_list()
    sbatch = src_shape[0]
    tbatch = tar_shape[0]
    ndepth = np.prod(src_shape[1:])

    src_new = optimal(tf.reshape(src_proj, [-1, ndepth]), tf.reshape(tar_proj, [-1, ndepth]), sbatch, tbatch, ndepth, nbins, alpha)
    src_new = tf.reshape(src_new, src_proj.get_shape())
    weight_t = tf.reshape(tf.transpose(wei), [fz,fz,3,f_num])
    src_out = tf.nn.conv2d_transpose(src_new, tf.transpose(weight), source.get_shape(), strides=[1, fz, fz, 1], padding='VALID')

    return src_out



def sliced_l1(source, target, fz=3, nbins=32, alpha=-1, name='idt'):
    f_num = fz * fz * 3
    src_shape = source.get_shape().as_list()
    tar_shape = target.get_shape().as_list()
    sbatch = src_shape[0]
    tbatch = tar_shape[0]
    ndepth = np.prod(src_shape[1:])

    weight, _ = tf.qr(tf.random_normal([f_num, f_num]))
    weight = tf.reshape(weight, [fz,fz,3,f_num])
    src_proj = tf.nn.relu(tf.nn.conv2d(source, weight, strides=[1, 1, 1, 1], padding='SAME'))
    tar_proj = tf.nn.relu(tf.nn.conv2d(target, weight, strides=[1, 1, 1, 1], padding='SAME'))


    src = tf.nn.max_pool(src_proj, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")
    tar = tf.nn.max_pool(tar_proj, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")

    # weight1, _ = tf.qr(tf.random_normal([f_num, f_num]))
    # weight1 = tf.reshape(weight1, [1, 1, f_num, f_num])
    # src = tf.nn.elu(tf.nn.conv2d(src, weight1, strides=[1, 1, 1, 1], padding='SAME'))
    # tar = tf.nn.elu(tf.nn.conv2d(tar, weight1, strides=[1, 1, 1, 1], padding='SAME'))

    sbat = sbatch
    tbat = tbatch
    # ndepth = 64
    # return ID_wasserstein(src_proj, tar_proj, sbat, tbat, ndepth, nbins, alpha)
    # return wasser_1d(src_proj, tar_proj, sbat, tbat, ndepth, nbins, alpha)
    return tf.reduce_mean(tf.abs(src - tar))

def wasser_1d(source, target, sbatch, tbatch, ndepth, nbins, alpha):
    # compute max min
    src_max = tf.reduce_max(source, reduction_indices=[0])
    src_min = tf.reduce_min(source, reduction_indices=[0])
    tar_max = tf.reduce_max(target, reduction_indices=[0])
    tar_min = tf.reduce_min(target, reduction_indices=[0])

    st_max = tf.maximum(src_max, tar_max)
    st_min = tf.minimum(src_min, tar_min)

    # compute histogram
    src_scal = (source - st_min) / (st_max - st_min) * nbins
    tar_scal = (target - st_min) / (st_max - st_min) * nbins

    src_hist, _ = my_hist.hist(src_scal, nbins, sbatch, ndepth, alpha)
    tar_hist, _ = my_hist.hist(tar_scal, nbins, tbatch, ndepth, alpha)

    # compute cumsum
    src_cumsum = tf.cumsum(src_hist, 0) / sbatch
    tar_cumsum = tf.cumsum(tar_hist, 0) / tbatch

    return tf.reduce_mean(tf.abs(src_cumsum - tar_cumsum))

def sliced_l2(source, target, size=16, nbins=32, alpha=-1, name='idt'):
    # target = tf.image.resize_nearest_neighbor(target, size=(size, size))
    # source = tf.image.resize_nearest_neighbor(source, size=(size, size))

    src_shape = source.get_shape().as_list()
    tar_shape = target.get_shape().as_list()
    sbatch = src_shape[0]
    tbatch = tar_shape[0]
    ndepth = np.prod(src_shape[1:])

    weight, _= tf.qr(tf.random_normal([ndepth, ndepth]))
    src_proj = tf.matmul(tf.reshape(source, [-1, ndepth]), weight)
    tar_proj = tf.matmul(tf.reshape(target, [-1, ndepth]), weight)

    return wasser_1d(src_proj, tar_proj, sbatch, tbatch, ndepth, nbins, alpha)

def sliced_l3(source, target, size=16, nbins=32, alpha=-1, name='idt'):
    src_shape = source.get_shape().as_list()
    tar_shape = target.get_shape().as_list()
    sbatch = src_shape[0]
    tbatch = tar_shape[0]

    height = src_shape[1]
    width = src_shape[2]

    w0, _ = tf.qr(tf.random_normal([height, height], mean=0.0, stddev=1.0))
    w1, _ = tf.qr(tf.random_normal([width * 3, width * 3], mean=0.0, stddev=1.0))

    w0_ = tf.expand_dims(w0, 0)
    w1_ = tf.expand_dims(w1, 0)

    w0_s = tf.tile(w0_, [sbatch, 1, 1])
    w1_s = tf.tile(w1_, [sbatch, 1, 1])
    src = tf.reshape(source, [sbatch, height, width * 3])  # NOTE!!!!!!!!!!!!!!!!
    src_proj = tf.matmul(tf.matmul(w0_s, src), w1_s)

    w0_t = tf.tile(w0_, [tbatch, 1, 1])
    w1_t = tf.tile(w1_, [tbatch, 1, 1])
    tar = tf.reshape(target, [tbatch, height, width * 3])
    tar_proj = tf.matmul(tf.matmul(w0_t, tar), w1_t)

    source_proj = tf.reshape(src_proj, [sbatch, height * width * 3])
    target_proj = tf.reshape(tar_proj, [tbatch, height * width * 3])

    # src_out = optimal(source_proj, target_proj, sbatch, tbatch, height * width * 3, nbins, alpha)

    return wasser_1d(source_proj, target_proj, sbatch, tbatch, height * width * 3, nbins, alpha)

    # return tf.reshape(out, [sbatch, height, width, 3])

def idt_2d(x, filter_shape, bias=True, stride_c=1, stride_d=3, name="conv2d", padding='VALID'):
    initializer = tf.random_normal_initializer(0., 0.02)
    with tf.variable_scope(name):
        weight = tf.get_variable("weight", shape=filter_shape, initializer=initializer)
        z = tf.nn.conv2d(x, weight, [1, stride_c, stride_c, 1], padding=padding)

        if bias:
            b = tf.get_variable("bias", shape=filter_shape[-1], initializer=tf.constant_initializer(0.))
            z = tf.nn.bias_add(z, b)
            z = tf.nn.elu(z)
            z = tf.nn.bias_add(z, -b)
        else:
            z = tf.nn.elu(z)

        w_tmp = tf.matrix_inverse(tf.transpose(tf.reshape(weight, [filter_shape[-1], filter_shape[-1]])))
        w_inv = tf.reshape(w_tmp, filter_shape)
        # print filter_shape
        if stride_c == stride_d:
            conv_stride = z
            deconv = tf.nn.conv2d_transpose(conv_stride, w_inv, x.get_shape(), strides=[1, stride_d, stride_d, 1],
                                            padding=padding)
        else:
            x_sh = x.get_shape().as_list()
            deconv = tf.zeros(x.get_shape())
            for i in xrange(stride_d):
                for j in xrange(stride_d):
                    conv_stride = tf.strided_slice(z, [0, i, j, 0], z.get_shape(), [1, stride_d, stride_d, 1])
                    tmp = tf.nn.conv2d_transpose(conv_stride, w_inv, [x_sh[0], x_sh[1] - i, x_sh[2] - j, x_sh[3]],
                                                    strides=[1, stride_d, stride_d, 1],
                                                    padding='VALID')
                    deconv += tf.pad(tmp, [[0, 0], [i, 0],[j, 0], [0, 0]])
            deconv /= stride_d * stride_d

    return deconv

def sink_horn(source, target, eps=0.1, loop=20):
    src_shape = source.get_shape().as_list()
    tar_shape = target.get_shape().as_list()
    sbatch = src_shape[0]
    tbatch = tar_shape[0]
    ndepth = np.prod(src_shape[1:])
    src = tf.reshape(source, [sbatch, ndepth])
    tar = tf.reshape(target, [tbatch, ndepth])

    ab = tf.matmul(src, tf.transpose(tar))
    a_2 = tf.diag_part(tf.matmul(src, tf.transpose(src)))
    a_2 = tf.tile(tf.expand_dims(a_2, -1), [1, tbatch])
    b_2 = tf.diag_part(tf.matmul(tar, tf.transpose(tar)))
    b_2 = tf.tile(tf.expand_dims(b_2, 0), [sbatch, 1])
    C = (a_2 - 2 * ab + b_2) / ndepth
    K = tf.exp(- C / eps)

    A = tf.expand_dims((1 / (tbatch * tf.reduce_sum(K, 1))),-1)
    B = 1 / (tbatch * tf.matmul(tf.transpose(K), A))
    for l in xrange(loop):
        A = 1 / (tbatch * tf.matmul(K, B))
        B = 1 / (tbatch * tf.matmul(tf.transpose(K), A))


    out = tf.matmul(tf.transpose(A), tf.matmul(tf.multiply(K, C), B))
    print out.get_shape().as_list()
    return tf.squeeze(out)

def idt_vec1(source, target, nbins=32, alpha=-1, name='idt'):
    src_shape = source.get_shape().as_list()
    tar_shape = target.get_shape().as_list()
    sbatch = src_shape[0]
    tbatch = tar_shape[0]
    ndepth = np.prod(src_shape[1:])

    initializer, _= tf.qr(tf.random_normal([ndepth, ndepth]))
    with tf.variable_scope(name):
        weight = tf.get_variable('w_proj', initializer=initializer)
        src_proj = tf.matmul(tf.reshape(source, [-1, ndepth]), weight)
        tar_proj = tf.matmul(tf.reshape(target, [-1, ndepth]), weight)

        src_new = optimal(src_proj, tar_proj, sbatch, tbatch, ndepth, nbins, alpha)
        src_out = tf.matmul(src_new, tf.transpose(weight))
    return tf.reshape(src_out, source.get_shape())

def idt_vec(source, target, nbins=32, alpha=-1, name='idt'):
    src_shape = source.get_shape().as_list()
    tar_shape = target.get_shape().as_list()
    sbatch = src_shape[0]
    tbatch = tar_shape[0]
    ndepth = np.prod(src_shape[1:])

    initializer, _= tf.qr(tf.random_normal([ndepth, ndepth]))
    with tf.variable_scope(name):
        weight = tf.get_variable('w_proj', initializer=initializer)
        src_proj = tf.matmul(source, weight)
        tar_proj = tf.matmul(target, weight)

        src_new = optimal(src_proj, tar_proj, sbatch, tbatch, ndepth, nbins, alpha)
        src_out = tf.matmul(src_new, tf.transpose(weight))
    return src_out

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
    tar_gat12 = my_gat.gat(tar_cumsum, idx12, tf.constant(nbins + 1), tf.constant(nbins + 1),
                           ndepth)
    tar_gat11 = my_gat.gat(tar_cumsum, idx11, tf.constant(nbins + 1), tf.constant(nbins + 1),
                           ndepth)
    f_prime = tar_gat12 - tar_gat11 + 1e-3
    f = 1.0 / f_prime * (src_cumsum - tar_gat11) + nn_idx1

    # g = interp1(u, f', D0R(i,:)); (corresponding to matlab code from original author)
    f_gat12 = my_gat.gat(f, idx22, tf.constant(nbins + 1), tf.constant(sbatch), ndepth)
    f_gat11 = my_gat.gat(f, idx21, tf.constant(nbins + 1), tf.constant(sbatch), ndepth)
    g_prime = f_gat12 - f_gat11
    g = g_prime * (src_scal - nn_idx2) + f_gat11 - 1

    src_new = g * (tar_max - tar_min) / nbins + tar_min
    return src_new

def idt_vec0(source, target, nbins=32, alpha=-1, name='idt'):
    src_shape = source.get_shape().as_list()
    tar_shape = target.get_shape().as_list()
    sbatch = src_shape[0]
    tbatch = tar_shape[0]
    ndepth = np.prod(src_shape[1:])

    initializer, _= tf.qr(tf.random_normal([ndepth, ndepth]))
    with tf.variable_scope(name):
        weight = tf.get_variable('w_proj', initializer=initializer)
        src_proj = tf.nn.tanh(tf.matmul(tf.reshape(source, [-1, ndepth]), weight))
        tar_proj = tf.nn.tanh(tf.matmul(tf.reshape(target, [-1, ndepth]), weight))

        src_new = optimal0(src_proj, tar_proj, sbatch, tbatch, ndepth, nbins, alpha)
        src_new = 0.5 * (tf.log(1+src_new)-tf.log(1-src_new))
        src_out = tf.matmul(src_new, tf.transpose(weight))
    return tf.reshape(src_out, source.get_shape())

def optimal0(source, target, sbatch, tbatch, ndepth, nbins, alpha):
    # compute max min
    # src_max = tf.reduce_max(source, reduction_indices=[0])
    # src_min = tf.reduce_min(source, reduction_indices=[0])
    # tar_max = tf.reduce_max(target, reduction_indices=[0])
    # tar_min = tf.reduce_min(target, reduction_indices=[0])

    # compute histogram
    src_scal = (source + 1) / 2. * nbins
    tar_scal = (target + 1) / 2. * nbins

    src_hist, _ = my_hist.hist(src_scal, nbins, sbatch, ndepth, alpha)
    tar_hist, _ = my_hist.hist(tar_scal, nbins, tbatch, ndepth, alpha)

    # compute cumsum
    src_cumsum = tf.cumsum(src_hist, 0) / sbatch
    tar_cumsum = tf.cumsum(tar_hist, 0) / tbatch

    # compute indices
    nn_idx1, idx11, idx12, nn_idx2, idx21, idx22 = my_ind.indices(src_scal, src_cumsum, tar_cumsum,
                                                                  sbatch, ndepth, nbins)
    # f = interp1(PY, 0:nbins, PX, 'linear'); (corresponding to matlab code from original author)
    tar_gat12 = my_gat.gat(tar_cumsum, idx12, tf.constant(nbins + 1), tf.constant(nbins + 1),
                           ndepth)
    tar_gat11 = my_gat.gat(tar_cumsum, idx11, tf.constant(nbins + 1), tf.constant(nbins + 1),
                           ndepth)
    f_prime = tar_gat12 - tar_gat11 + 1e-3
    f = 1.0 / f_prime * (src_cumsum - tar_gat11) + nn_idx1

    # g = interp1(u, f', D0R(i,:)); (corresponding to matlab code from original author)
    f_gat12 = my_gat.gat(f, idx22, tf.constant(nbins + 1), tf.constant(sbatch), ndepth)
    f_gat11 = my_gat.gat(f, idx21, tf.constant(nbins + 1), tf.constant(sbatch), ndepth)
    g_prime = f_gat12 - f_gat11
    g = g_prime * (src_scal - nn_idx2) + f_gat11 - 1

    src_new = g * 2 / nbins - 1.
    return src_new

def resize(x, shape):
    h = shape[1]
    w = shape[2]
    return tf.image.resize_nearest_neighbor(x, size=(h, w))

def conv_block(x, y, f_x, f_y,  bias=True, stride=1, name="conv2d", padding='VALID'):
    initializer = tf.random_normal_initializer(0., 0.02)
    with tf.variable_scope(name):
        w_x = tf.get_variable("weight", shape=f_x, initializer=initializer)
        x = tf.nn.conv2d(x, w_x, [1, stride, stride, 1], padding=padding)

        if f_x == f_y:
            w_y = w_x
        else:
            w_y = tf.reduce_mean(tf.reshape(w_x, [f_y[0], f_y[1], f_y[2], f_y[3], -1]), -1)
        print w_y
        y = tf.nn.conv2d(y, w_y, [1, stride, stride, 1], padding=padding)

        if bias:
            b = tf.get_variable("bias", shape=f_x[-1], initializer=tf.constant_initializer(0.))
            x = tf.nn.bias_add(x, b)
            y = tf.nn.bias_add(y, b)
    return tf.nn.elu(x), tf.nn.elu(y)

def deconv_block(y, f_x, f_y,  output_shape, bias=True, stride=3, is_stride=True, name="conv2d", padding='VALID'):
    y = ielu(y)
    initializer = tf.random_normal_initializer(0., 0.02)
    with tf.variable_scope(name):
        if bias:
            b = tf.get_variable("bias", shape=f_x[-1], initializer=tf.constant_initializer(0.))
            y = tf.nn.bias_add(y, -b)

        w_x = tf.get_variable("weight", shape=f_x, initializer=initializer)
        if f_x == f_y:
            w_y = w_x
        else:
            w_y = tf.reduce_mean(tf.reshape(w_x, [f_y[0], f_y[1], f_y[2], f_y[3], -1]), -1)

        w_tmp = tf.matrix_inverse(tf.transpose(tf.reshape(w_y, [f_y[-1], f_y[-1]])))
        w_inv = tf.reshape(w_tmp, f_y)
        if is_stride:
            y_stride = tf.strided_slice(y, [0, 0, 0, 0], y.get_shape(), [1, stride, stride, 1])
        else:
            y_stride = y
        deconv = tf.nn.conv2d_transpose(y_stride, w_inv, output_shape, strides=[1, stride, stride, 1],
                                        padding=padding)
    return deconv


def ielu(x):
    pos = tf.nn.relu(x)
    neg = tf.log((x - tf.abs(x)) * 0.5 + 1)
    return pos + neg

# def idt_vec(source, target, nbins=32, alpha=-1, name='idt'):
#     src_shape = source.get_shape().as_list()
#     tar_shape = target.get_shape().as_list()
#     sbatch = src_shape[0]
#     tbatch = tar_shape[0]
#     ndepth = np.prod(src_shape[1:])
#
#     weight, _= tf.qr(tf.random_normal([ndepth, ndepth], seed=1))
#
#     # weight = tf.get_variable('w_proj', initializer=initializer)
#     src_proj = tf.matmul(tf.reshape(source, [-1, ndepth]), weight)
#     tar_proj = tf.matmul(tf.reshape(target, [-1, ndepth]), weight)
#
#     src_new = optimal(src_proj, tar_proj, sbatch, tbatch, ndepth, nbins, alpha)
#     src_out = tf.matmul(src_new, tf.transpose(weight))
#     return tf.reshape(src_out, source.get_shape())
#
# def idt_one(source, nbins=32, alpha=-1, name='idt'):
#     src_shape = source.get_shape().as_list()
#     sbatch = src_shape[0]
#     ndepth = np.prod(src_shape[1:])
#
#     # weight = tf.get_variable('w_proj', initializer=initializer)
#     src_proj = tf.matmul(tf.reshape(source, [-1, ndepth]), weight)
#     tar_proj = tf.matmul(tf.reshape(target, [-1, ndepth]), weight)
#
#     src_new = optimal(src_proj, tar_proj, sbatch, tbatch, ndepth, nbins, alpha)
#     src_out = tf.matmul(src_new, tf.transpose(weight))
#     return tf.reshape(src_out, source.get_shape())

def optimal_1d(source, sbatch, ndepth, nbins = 32, alpha = -1):
    # compute max min
    src_max = tf.reduce_max(source, reduction_indices=[0])
    src_min = tf.reduce_min(source, reduction_indices=[0])
    # tar_max = tf.reduce_max(target, reduction_indices=[0])
    # tar_min = tf.reduce_min(target, reduction_indices=[0])

    # compute histogram
    src_scal = (source + 1) / 2 * nbins
    # tar_scal = (target - tar_min) / (tar_max - tar_min) * nbins

    src_hist, _ = my_hist.hist(src_scal, nbins, sbatch, ndepth, alpha)
    # tar_hist, _ = my_hist.hist(tar_scal, nbins, tbatch, ndepth, alpha)

    # compute cumsum
    src_cumsum = tf.cumsum(src_hist, 0) / sbatch
    # tar_cumsum = tf.cumsum(tar_hist, 0) / tbatch

    # compute indices
    _, _, _, nn_idx2, idx21, idx22 = my_ind.indices(src_scal, src_cumsum, src_cumsum,
                                                                   sbatch, ndepth, nbins)
    # f = interp1(PY, 0:nbins, PX, 'linear'); (corresponding to matlab code from original author)
    # tar_gat12 = my_gat.gat(tar_cumsum, idx12, tf.constant(nbins + 1), tf.constant(nbins + 1),
    #                        ndepth)
    # tar_gat11 = my_gat.gat(tar_cumsum, idx11, tf.constant(nbins + 1), tf.constant(nbins + 1),
    #                        ndepth)
    # f_prime = tar_gat12 - tar_gat11 + 1e-3
    # f = 1.0 / f_prime * (src_cumsum - tar_gat11) + nn_idx1

    # g = interp1(u, f', D0R(i,:)); (corresponding to matlab code from original author)
    f_gat12 = my_gat.gat(src_cumsum, idx22, tf.constant(nbins + 1), tf.constant(sbatch), ndepth)
    f_gat11 = my_gat.gat(src_cumsum, idx21, tf.constant(nbins + 1), tf.constant(sbatch), ndepth)
    g_prime = f_gat12 - f_gat11
    g = g_prime * (src_scal - nn_idx2) + f_gat11

    src_new = g * 2 - 1
    src_new = tf.minimum(tf.maximum(src_new, -1), 1)
    return src_new