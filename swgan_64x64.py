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
import tflib.plot
import tflib.ops.layers6

import fid
import re

sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']


# flags = tf.app.flags
#
# flags.DEFINE_integer("FEATURE_DIM", 128, "feature dimension")
# flags.DEFINE_integer("LAMBDA1", 20, "lambda1")
# flags.DEFINE_integer("LAMBDA2", 10, "lambda2")
# flags.DEFINE_integer("BLOCK_NUM", 3, "dual block number")
#
#
# FLAGS = flags.FLAGS



DATA_DIR = '/usr/bendernas02/bendercache-a/celeba/64_crop'
DATASET = "celeba"  # celeba, cifar10, svhn, lsun
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

# Download the Inception model from here
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# And set the path to the extracted model here:
INCEPTION_DIR = "inception-2015-12-05"

# Path to the real world statistics file.
STAT_FILE = "stats/fid_stats_celeba.npz"

MODE = 'swgan'  # dcgan, wgan, wgan-gp, lsgan
DIM = 64  # Model dimensionality

LOAD_CHECKPOINT = False


CRITIC_ITERS = 4  # How many iterations to train the critic for
D_LR = 0.0003
G_LR = 0.0003
BETA1_D = 0.
BETA1_G = 0.
FID_STEP = 2000  # FID evaluation every FID_STEP
ITERS = 100000  # How many iterations to train for

OUTPUT_STEP = 200  # Print output every OUTPUT_STEP
SAVE_SAMPLES_STEP = 200  # Generate and save samples every SAVE_SAMPLES_STEP

DIR = "mmdd_hhmmss_lrd_lrg"
ITER_START = 0

# Switch on and off batchnormalizaton for the discriminator
# and the generator. Default is on for both.
BN_D = True
BN_G = True

# Log subdirectories are automatically created from
# the above settings and the current timestamp.
CHECKPOINT_STEP = 5000  # FID_STEP
LOG_DIR = "/srv/glusterfs/jwu/logs_cvpr19"  # Directory for Tensorboard events, checkpoints and samples
N_GPUS = 1  # Number of GPUs
BATCH_SIZE = 64  # Batch size. Must be a multiple of N_GPUS
LAMBDA = 10  # Gradient penalty lambda hyperparameter
OUTPUT_DIM = DIM * DIM * 3  # Number of pixels in each iamge

FEATURE_DIM = 128
LAMBDA1 = 20
LAMBDA2 = 10

if not LOAD_CHECKPOINT:
    timestamp = time.strftime("%m%d_%H%M%S")
    DIR = "baseline_swgan"
else:
    DIR = "baseline_swgan"

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

# FID evaluation.
FID_EVAL_SIZE = 50000  # Number of samples for evaluation
FID_SAMPLE_BATCH_SIZE = 1000  # Batch size of generating samples, lower to save GPU memory
FID_BATCH_SIZE = 200  # Batch size for final FID calculation i.e. inception propagation etc.

# Load checkpoint
# from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
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


# lib.print_model_settings(locals().copy(), LOG_DIR)

def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """

    # For actually generating decent samples, use this one
    return GoodGenerator, GoodDiscriminator
    raise Exception('You must choose an architecture!')


DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = tflib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = tflib.ops.linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)


def Normalize(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'swgan'):
        if axes != [0, 2, 3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return tflib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs)
    else:
        # return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
        return tflib.ops.batchnorm.Batchnorm(name, axes, inputs, fused=True)


def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)


def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4 * kwargs['output_dim']
    output = tflib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    return output


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = tflib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    output = tflib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = tflib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def BottleneckResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = functools.partial(tflib.ops.conv2d.Conv2D, stride=2)
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim // 2, output_dim=output_dim // 2,
                                    stride=2)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=output_dim // 2, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = SubpixelConv2D
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b = functools.partial(tflib.ops.deconv2d.Deconv2D, input_dim=input_dim // 2, output_dim=output_dim // 2)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=output_dim // 2, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = tflib.ops.conv2d.Conv2D
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim // 2, output_dim=output_dim // 2)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim // 2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name + '.Conv1', filter_size=1, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_1b(name + '.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_2(name + '.Conv2', filter_size=1, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name + '.BN', [0, 2, 3], output)

    return shortcut + (0.3 * output)


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True, bn=False):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = MeanPoolConv
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = UpsampleConv
        conv_1 = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = tflib.ops.conv2d.Conv2D
        conv_1 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(tflib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    if bn:
        output = Normalize(name + '.BN1', [0, 2, 3], output)
    output = tf.nn.relu(output)
    output = conv_1(name + '.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    if bn:
        output = Normalize(name + '.BN2', [0, 2, 3], output)
    output = tf.nn.relu(output)
    output = conv_2(name + '.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


def SWDBlock(name, inputs):
    i_sh = inputs.get_shape().as_list()
    with tf.name_scope(name) as scope:
        off = lib.param(name + '.offset', np.zeros((1, i_sh[1]), dtype='float32'))
        sca = lib.param(name + '.scale', np.ones((1, i_sh[1]), dtype='float32'))
        offset = tf.tile(off, [i_sh[0], 1])
        scale = tf.tile(sca, [i_sh[0], 1])
        output = LeakyReLU(tf.multiply(scale, inputs) + offset)
        return output, scale


# ! Generators
def GoodGenerator(n_samples, noise=None, dim=DIM, nonlinearity=tf.nn.relu, bn=BN_G):
    if noise is None:
        noise = tf.random_normal([n_samples, FEATURE_DIM])

    ## supports 32x32 images
    fact = DIM // 16

    output = lib.ops.linear.Linear('Generator.Input', FEATURE_DIM, fact*fact*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, fact, fact])
    output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up', bn=bn)
    output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up', bn=bn)
    output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up', bn=bn)
    output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up', bn=bn)
    if bn:
        output = Normalize('Generator.OutputN', [0,2,3], output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])


# ! Discriminators
def GoodDiscriminator(inputs, dim=DIM, bn=BN_D):
    i_sh = inputs.get_shape().as_list()
    output = tf.reshape(inputs, [-1, 3, DIM, DIM])
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)

    output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, resample='down', bn=bn)
    output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, resample='down', bn=bn)
    output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, resample='down', bn=bn)
    output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, resample='down', bn=bn)

    output = tf.reshape(output, [i_sh[0], -1])
    o_sh = output.get_shape().as_list()
    latent_dim = FEATURE_DIM
    output = tflib.ops.linear.Linear('Discriminator.Output', o_sh[1], latent_dim, output)
    return output

def SWD(inputs):
    i_sh = inputs.get_shape().as_list()
    name = 'Discriminator.SWD'
    with tf.name_scope(name) as scope:
        weight_value0= tf.random_normal([i_sh[1], i_sh[1]])
        weight0 = lib.param(name+'.SWD_proj0', weight_value0)
        proj0 = tf.matmul(tf.reshape(inputs, [-1, i_sh[1]]), weight0)
        output0, scale0 = SWDBlock('Discriminator.idt0', proj0)

        weight_value1= tf.random_normal([i_sh[1], i_sh[1]])
        weight1 = lib.param(name+'.SWD_proj1', weight_value1)
        proj1 = tf.matmul(tf.reshape(inputs, [-1, i_sh[1]]), weight1)
        output1, scale1 = SWDBlock('Discriminator.idt1', proj1)

        weight_value2= tf.random_normal([i_sh[1], i_sh[1]])
        weight2 = lib.param(name+'.SWD_proj2', weight_value2)
        proj2 = tf.matmul(tf.reshape(inputs, [-1, i_sh[1]]), weight2)
        output2, scale2 = SWDBlock('Discriminator.idt2', proj2)

        weight_value3 = tf.random_normal([i_sh[1], i_sh[1]])
        weight3 = lib.param(name + '.SWD_proj3', weight_value3)
        proj3 = tf.matmul(tf.reshape(inputs, [-1, i_sh[1]]), weight3)
        output3, scale3 = SWDBlock('Discriminator.idt3', proj3)

        output = tf.concat([output0, output1, output2, output3], 1)
        scale = tf.concat([scale0, scale1, scale2, scale3], 1)

    return output, scale


Generator, Discriminator = GeneratorAndDiscriminator()


def train_model():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, DIM, DIM])
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
        gen_costs, disc_costs = [], []

        for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
            with tf.device(device):

                real_data = tf.reshape(2 * ((tf.cast(real_data_conv, tf.float32) / 255.) - .5),
                                       [BATCH_SIZE // len(DEVICES), OUTPUT_DIM])
                fake_data = Generator(BATCH_SIZE // len(DEVICES), bn=BN_G)

                r_feat = Discriminator(real_data)
                f_feat = Discriminator(fake_data)
                disc_real = tf.reduce_mean(SWD(r_feat)[0], [1])
                disc_fake = tf.reduce_mean(SWD(f_feat)[0], [1])

                if MODE == 'swgan':
                    gen_cost = tf.reduce_mean(disc_fake)
                    disc_cost = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
                    alpha = tf.random_uniform(
                        shape=[BATCH_SIZE // len(DEVICES), 1],
                        minval=0.,
                        maxval=1.
                    )
                    # differences = fake_data - real_data
                    interpolates = (1 - alpha) * real_data + alpha * fake_data
                    gradients = (1 - alpha) * \
                                tf.gradients(tf.reduce_mean(SWD(Discriminator(interpolates, bn=BN_D))[0], [1]), interpolates)[0]
                    slopes = tf.reduce_sum(tf.square(gradients), reduction_indices=[1])
                    gradient_penalty = tf.reduce_mean(slopes)

                    alpha1 = tf.random_uniform(
                        shape=[BATCH_SIZE // len(DEVICES), 1],
                        minval=0.,
                        maxval=1.
                    )
                    inter_feat = (1 - alpha1) * r_feat + alpha1 * f_feat
                    lrelu, scale = SWD(inter_feat)

                    #directly compute the derivate of swd block
                    grad_feat = tf.multiply(scale, 0.1*(tf.abs(lrelu) - lrelu) + 0.5*(tf.abs(lrelu) + lrelu))

                    pen_feat = tf.reduce_mean(tf.square(grad_feat - 0.001))
                    disc_cost += LAMBDA1 * (gradient_penalty) + LAMBDA2 * pen_feat
                else:
                    raise Exception()

                gen_costs.append(gen_cost)
                disc_costs.append(disc_cost)

        gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        disc_cost = tf.add_n(disc_costs) / len(DEVICES)

        if MODE == 'swgan':
            gen_train_op = tf.train.AdamOptimizer(learning_rate=G_LR, beta1=BETA1_G, beta2=0.9).minimize(gen_cost,
                                                                                                         var_list=lib.params_with_name(
                                                                                                             'Generator'),
                                                                                                         colocate_gradients_with_ops=True)

            optimizer = tf.train.AdamOptimizer(learning_rate=D_LR, beta1=BETA1_D, beta2=0.9)
            grads_and_vars = optimizer.compute_gradients(disc_cost,
                                                         var_list=lib.params_with_name('Discriminator'))

            for idx, (egrad, var) in enumerate(grads_and_vars):
                # print var.name
                if 'SWD_proj' in var.name:
                    # print var.name
                    tmp1 = tf.matmul(tf.transpose(var), egrad)
                    tmp2 = 0.5 * (tmp1 + tf.transpose(tmp1))
                    rgrad = egrad - tf.matmul(var, tmp2)
                    grads_and_vars[idx] = (rgrad, var)
            disc_train_op = optimizer.apply_gradients(grads_and_vars)

            # stiefel update
            stiefel_up = tf.random_normal([FEATURE_DIM, FEATURE_DIM])
            for var in lib.params_with_name('Discriminator.SWD'):
                # print var.name
                if 'SWD_proj' in var.name:
                    print(var.name)
                    o_n, _ = tf.qr(var)
                    stiefel_up = stiefel_up + tf.reduce_sum(var.assign(o_n), [0, 1])
        else:
            raise Exception()

        tf.summary.scalar("gen_cost", gen_cost)
        tf.summary.scalar("disc_cost", disc_cost)

        summary_op = tf.summary.merge_all()

        # For generating samples
        fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, FEATURE_DIM)).astype('float32'))
        all_fixed_noise_samples = []
        for device_index, device in enumerate(DEVICES):
            n_samples = BATCH_SIZE // len(DEVICES)
            all_fixed_noise_samples.append(Generator(n_samples,
                                                     noise=fixed_noise[
                                                           device_index * n_samples:(device_index + 1) * n_samples]))
        if tf.__version__.startswith('1.'):
            all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
        else:
            all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)

        def generate_image(iteration):
            samples = session.run(all_fixed_noise_samples)
            samples = ((samples + 1.) * (255.99 // 2)).astype('int32')
            tflib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, DIM, DIM)),
                                          '%s/samples_%d.png' % (SAMPLES_DIR, iteration))

        fid_tfvar = tf.Variable(0.0, trainable=False)
        fid_sum = tf.summary.scalar("FID", fid_tfvar)
        writer = tf.summary.FileWriter(TBOARD_DIR, session.graph)

        # Dataset iterator
        train_gen, dev_gen = tflib.data_loader.load(BATCH_SIZE, DATA_DIR, DATASET)

        def inf_train_gen():
            while True:
                for (images,) in train_gen():
                    yield images

        # Save a batch of ground-truth samples
        _x = inf_train_gen().__next__()
        _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:BATCH_SIZE // N_GPUS]})
        _x_r = ((_x_r + 1.) * (255.99 // 2)).astype('int32')
        tflib.save_images.save_images(_x_r.reshape((BATCH_SIZE // N_GPUS, 3, DIM, DIM)),
                                      '%s/samples_groundtruth.png' % SAMPLES_DIR)
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

        # load model
        print("load inception model..")

        fid.create_inception_graph(os.path.join(INCEPTION_DIR, "classify_image_graph_def.pb"))
        print("ok")
        
        print("load train stats.. ")

        # load precalculated training set statistics
        f = np.load(STAT_FILE)
        mu_real, sigma_real = f['mu'][:], f['sigma'][:]
        f.close()
        print("ok")

        # Train loop

        for it in range(ITERS):
            iteration = it

            start_time = time.time()

            # Train generator
            if iteration > 0:
                _data = gen.__next__()

                _gen_cost, _ = session.run([gen_cost, gen_train_op], feed_dict={all_real_data_conv: _data})

            # Train critic
            for i in range(CRITIC_ITERS):
                _data = gen.__next__()
                _disc_cost, _, _, _summary_op = session.run([disc_cost, disc_train_op, stiefel_up, summary_op],
                                                            feed_dict={all_real_data_conv: _data})

            if iteration % SAVE_SAMPLES_STEP == SAVE_SAMPLES_STEP - 1:
                generate_image(iteration)
                print("Time: %g/itr, Itr: %d, generator loss: %g , discriminator_loss: %g" % (
                    time.time() - start_time, iteration, _gen_cost, _disc_cost))
                writer.add_summary(_summary_op, iteration)

            if iteration % FID_STEP == FID_STEP - 1 and iteration >= 9999:
                # FID
                samples = np.zeros((FID_EVAL_SIZE, OUTPUT_DIM), dtype=np.uint8)

                n_fid_batches = FID_EVAL_SIZE // FID_SAMPLE_BATCH_SIZE

                for i in range(n_fid_batches):
                    frm = i * FID_SAMPLE_BATCH_SIZE
                    to = frm + FID_SAMPLE_BATCH_SIZE
                    tmp = session.run(Generator(FID_SAMPLE_BATCH_SIZE))
                    samples[frm:to] = ((tmp + 1.0) * 127.5).astype('uint8')

                # Cast, reshape and transpose (BCHW -> BHWC)
                samples = samples.reshape(FID_EVAL_SIZE, 3, DIM, DIM)
                samples = samples.transpose(0, 2, 3, 1)

                print("ok")

                mu_gen, sigma_gen = fid.calculate_activation_statistics(samples,
                                                                        session,
                                                                        batch_size=FID_BATCH_SIZE,
                                                                        verbose=True)

                try:
                    FID = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
                except Exception as e:
                    print(e)
                    FID = 500

                print("calculate FID: %d " % (FID))

                # print(FID)

                session.run(tf.assign(fid_tfvar, FID))
                summary_str = session.run(fid_sum)
                writer.add_summary(summary_str, iteration)

            # Save checkpoint
            if iteration % CHECKPOINT_STEP == CHECKPOINT_STEP - 1:
                if iteration == CHECKPOINT_STEP - 1:
                    ckpt_saver.save(session,
                                    os.path.join(CHECKPOINT_DIR, "swgan.model"),
                                    iteration, write_meta_graph=True)
                else:
                    ckpt_saver.save(session,
                                    os.path.join(CHECKPOINT_DIR, "swgan.model"),
                                    iteration, write_meta_graph=False)


if __name__ == "__main__":
    train_model()

