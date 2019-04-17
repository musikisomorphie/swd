import os
import sys
import time
import functools

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.data_loader
import tflib.ops.layernorm
import tflib.ops.layers6
import tflib.plot
import re

sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']

DATA_DIR = ""
LOG_DIR = "" # Directory for Tensorboard events, checkpoints and samples
DIR = ""

DATASET = "celeba" # celeba, cifar10, svhn, lsun
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

MODE = 'swae' # dcgan, wgan, wgan-gp, lsgan
DIM = 64 # Model dimensionality
LOAD_CHECKPOINT = True
D_LR = 0.0003
G_LR = 0.0003
BETA1_D = 0.0
BETA1_G = 0.0
ITERS = 200000  # How many iterations to train for
OUTPUT_STEP = 400 # Print output every OUTPUT_STEP
SAVE_SAMPLES_STEP = 400 # Generate and save samples every SAVE_SAMPLES_STEP

ITER_START = 0

# Switch on and off batchnormalizaton for the discriminator
# and the generator. Default is on for both.
BN_D=True
BN_G=True

# Log subdirectories are automatically created from
# the above settings and the current timestamp.
CHECKPOINT_STEP = 5000
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = DIM * DIM * 3 # Number of pixels in each iamge
FEATURE_DIM = 128
BIN = 32

LOG_DIR = os.path.join(LOG_DIR, DIR)
SAMPLES_DIR = os.path.join(LOG_DIR, "samples")
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")
TBOARD_DIR = os.path.join(LOG_DIR, "logs")

# Create directories if necessary
if not os.path.exists(SAMPLES_DIR):
  print("*** create sample dir %s" % SAMPLES_DIR)
  os.makedirs(SAMPLES_DIR)
if not os.path.exists(CHECKPOINT_DIR):
  print("*** create checkpoint dir %s" % CHECKPOINT_DIR)
  os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(TBOARD_DIR):
  print("*** create tboard dir %s" % TBOARD_DIR)
  os.makedirs(TBOARD_DIR)


# Load checkpoint
def load_checkpoint(session, saver, checkpoint_dir):
  print(" [*] Reading checkpoints...")
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  print(checkpoint_dir)
  i = 0

  if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))

    latest_cp = tf.train.latest_checkpoint(checkpoint_dir)

    i = int(re.findall('\d+', latest_cp)[-1]) + 1

    print(" [*] Success to read {}".format(ckpt_name))
    return True, i
  else:
    print(" [*] Failed to find a checkpoint")
    return False, i


DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

def Pool(x, r=2, s=1):
    output = tf.transpose(x, [0, 2, 3, 1], name='NCHW_to_NHWC')
    output = tf.nn.avg_pool(output, ksize=[1, r, r, 1], strides=[1, s, s, 1], padding="SAME")
    return tf.transpose(output, [0, 3, 1, 2], name='NHWC_to_NCHW')


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = tflib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = tflib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)


def Normalize(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'aae-wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return tflib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs)
    else:
        #return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
        return tflib.ops.batchnorm.Batchnorm(name, axes, inputs, fused=True)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = tflib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = tflib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = tflib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = tflib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def BottleneckResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(tflib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b       = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim // 2, output_dim=output_dim // 2, stride=2)
        conv_2        = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=output_dim // 2, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b       = functools.partial(tflib.ops.deconv2d.Deconv2D, input_dim=input_dim // 2, output_dim=output_dim // 2)
        conv_2        = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=output_dim // 2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = tflib.ops.conv2d.Conv2D
        conv_1        = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b       = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim // 2, output_dim=output_dim // 2)
        conv_2        = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim // 2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN', [0,2,3], output)

    return shortcut + (0.3*output)

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True, bn=False):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = tflib.ops.conv2d.Conv2D
        conv_1        = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    if bn:
      output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    if bn:
      output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output

# ! Generators
def NearestNeighbor(x, size):
    x = tf.transpose(x, [0, 2, 3, 1], name='NCHW_to_NHWC')
    x = tf.image.resize_nearest_neighbor(x, size=(size, size))
    x = tf.transpose(x, [0, 3, 1, 2], name='NHWC_to_NCHW')
    return x


def BEGANGenerator(n_samples, noise=None, dim=DIM, nonlinearity=tf.nn.relu, bn=BN_G):
    if noise is None:
        noise = tf.random_normal([n_samples, FEATURE_DIM])

    dim = 32

    output = tflib.ops.linear.Linear('Generator.Linear', FEATURE_DIM, 8 * 8 * dim, noise)
    output = tf.reshape(output, [-1, dim, 8, 8])

    output = tflib.ops.conv2d.Conv2D('Generator.Input1', dim, dim, 3, output, he_init=False)
    output = tf.nn.elu(output)
    output = tflib.ops.conv2d.Conv2D('Generator.Input2', dim, dim, 3, output, he_init=False)
    output = tf.nn.elu(output)

    output = NearestNeighbor(output, DIM // 4)
    output = tflib.ops.conv2d.Conv2D('Generator.Input3', dim, dim, 3, output, he_init=False)
    output = tf.nn.elu(output)
    output = tflib.ops.conv2d.Conv2D('Generator.Input4', dim, dim, 3, output, he_init=False)
    output = tf.nn.elu(output)

    output = NearestNeighbor(output, DIM // 2)
    output = tflib.ops.conv2d.Conv2D('Generator.Input5', dim, dim, 3, output, he_init=False)
    output = tf.nn.elu(output)
    output = tflib.ops.conv2d.Conv2D('Generator.Input6', dim, dim, 3, output, he_init=False)
    output = tf.nn.elu(output)

    output = NearestNeighbor(output, DIM)
    output = tflib.ops.conv2d.Conv2D('Generator.Input7', dim, dim, 3, output, he_init=False)
    output = tf.nn.elu(output)
    output = tflib.ops.conv2d.Conv2D('Generator.Input8', dim, dim, 3, output, he_init=False)
    output = tf.nn.elu(output)

    output = tflib.ops.conv2d.Conv2D('Generator.Input9', dim, 3, 3, output, he_init=False)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])


def BEGANEncoder(inputs, dim=DIM, bn=BN_D, n_samples=64, loop=4):
    i_sh = inputs.get_shape().as_list()
    dim = 32
    output = tf.reshape(inputs, [-1, 3, DIM, DIM])
    output = NearestNeighbor(output, 8)

    output = tf.reshape(output, [i_sh[0], -1])
    o_sh = output.get_shape().as_list()
    output = tflib.ops.linear.Linear('Encoder.Linear', o_sh[1], FEATURE_DIM, output)
    noise = tf.random_normal([i_sh[0], FEATURE_DIM])

    o_sh = output.get_shape().as_list()
    output = tflib.ops.layers6.SWDBlock('Encoder.swd', output, noise, nbins=BIN)
    output.set_shape(o_sh)
    output = tflib.ops.layers6.SWDBlock('Encoder.swd1', output, noise, nbins=BIN)
    output.set_shape(o_sh)
    output = tflib.ops.layers6.SWDBlock('Encoder.swd2', output, noise, nbins=BIN)
    output.set_shape(o_sh)

    return output


def train_model():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

        all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, DIM, DIM])
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
        gen_costs, disc_costs, recon_costs = [], [], []

        for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
            with tf.device(device):

                real_data = tf.reshape(2 * ((tf.cast(real_data_conv, tf.float32) / 255.) - .5),
                                       [BATCH_SIZE // len(DEVICES), OUTPUT_DIM])

                fake_z = BEGANEncoder(real_data)
                fake_data = BEGANGenerator(BATCH_SIZE // len(DEVICES), noise=fake_z, bn=BN_G)

                if MODE == 'swae':
                    recon_cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(real_data, fake_data)))
                else:
                    raise Exception()

                recon_costs.append(recon_cost)
        recon_cost = tf.add_n(recon_costs) / len(DEVICES)

        if MODE == 'swae':
            optimizer = tf.train.AdamOptimizer(learning_rate=D_LR, beta1=BETA1_D, beta2=0.9)

            grads_and_vars = optimizer.compute_gradients(recon_cost,
                                                         var_list=lib.params_with_name1('Encoder', 'Generator'))
            for idx, (egrad, var) in enumerate(grads_and_vars):
                # print var.name
                if 'swd_proj' in var.name:
                    # print var.name
                    tmp1 = tf.matmul(tf.transpose(var), egrad)
                    tmp2 = 0.5 * (tmp1 + tf.transpose(tmp1))
                    rgrad = egrad - tf.matmul(var, tmp2)
                    grads_and_vars[idx] = (rgrad, var)
            recon_train_op = optimizer.apply_gradients(grads_and_vars)

            # stiefel update
            stiefel_up = tf.random_normal([FEATURE_DIM, FEATURE_DIM])
            for var in lib.params_with_name('Encoder.swd'):
                # print var.name
                if 'swd_proj' in var.name:
                    print(var.name)

                    o_n, _ = tf.qr(var)
                    stiefel_up = stiefel_up + tf.reduce_sum(var.assign(o_n), [0, 1])

            tf.summary.scalar("recon_cost", recon_cost)
            summary_op = tf.summary.merge_all()
        else:
            raise Exception()

        # For generating samples
        fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, FEATURE_DIM)).astype('float32'))
        all_fixed_noise_samples = []
        for device_index, device in enumerate(DEVICES):
            n_samples = BATCH_SIZE // len(DEVICES)
            all_fixed_noise_samples.append(BEGANGenerator(n_samples))
        if tf.__version__.startswith('1.'):
            all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
        else:
            all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)

        def generate_image(iteration):
            samples = session.run(all_fixed_noise_samples)
            samples = ((samples + 1.) * 127.5).astype('int32')
            tflib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, DIM, DIM)),
                                          '%s/samples_%d.png' % (SAMPLES_DIR, iteration))

        writer = tf.summary.FileWriter(TBOARD_DIR, session.graph)

        # Dataset iterator
        train_gen, dev_gen = tflib.data_loader.load(BATCH_SIZE, DATA_DIR, DATASET)

        def inf_train_gen():
            while True:
                for (images,) in train_gen():
                    yield images

        # Save a batch of ground-truth samples
        _x = inf_train_gen().__next__()
        _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:BATCH_SIZE//N_GPUS]})
        _x_r = ((_x_r+0.5)*255).astype('int32')
        tflib.save_images.save_images(_x_r.reshape((BATCH_SIZE // N_GPUS, 3, DIM, DIM)), '%s/samples_groundtruth.png' % SAMPLES_DIR)

        session.run(tf.global_variables_initializer())

        # Checkpoint saver
        ckpt_saver = tf.train.Saver(max_to_keep=int(ITERS / CHECKPOINT_STEP))

        if LOAD_CHECKPOINT:
            is_check, ITER_START = load_checkpoint(session, ckpt_saver, CHECKPOINT_DIR)
            if is_check:
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        gen = inf_train_gen()

        for it in range(ITERS):
            iteration = it + ITER_START
            start_time = time.time()
            _data = gen.__next__()

            _recon_cost, _, _, _summary_op = session.run([recon_cost, recon_train_op, stiefel_up, summary_op],
                                                         feed_dict={all_real_data_conv: _data})

            writer.add_summary(_summary_op, iteration)

            if iteration % SAVE_SAMPLES_STEP == SAVE_SAMPLES_STEP - 1:
                generate_image(iteration)
                print("Time: %g/itr, Itr: %d, reconstruction loss: %g" % (
                    time.time() - start_time, iteration, _recon_cost))

            # Save checkpoint
            if (iteration != 0) and (iteration % CHECKPOINT_STEP == CHECKPOINT_STEP - 1):
                if iteration == CHECKPOINT_STEP - 1:
                    ckpt_saver.save(session,
                                    os.path.join(CHECKPOINT_DIR, "SWAE.model"),
                                    iteration, write_meta_graph=True)
                else:
                    ckpt_saver.save(session,
                                    os.path.join(CHECKPOINT_DIR, "SWAE.model"),
                                    iteration, write_meta_graph=False)


if __name__ == "__main__":
    train_model()
