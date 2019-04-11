import numpy as np

import os
import urllib
import gzip
import cPickle as pickle



# def rgb_to_hsv(rgb):
#     # """
#     # >>> from colorsys import rgb_to_hsv as rgb_to_hsv_single
#     # >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(50, 120, 239))
#     # 'h=0.60 s=0.79 v=239.00'
#     # >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(163, 200, 130))
#     # 'h=0.25 s=0.35 v=200.00'
#     # >>> np.set_printoptions(2)
#     # >>> rgb_to_hsv(np.array([[[50, 120, 239], [163, 200, 130]]]))
#     # array([[[   0.6 ,    0.79,  239.  ],
#     #         [   0.25,    0.35,  200.  ]]])
#     # >>> 'h={:.2f} s={:.2f} v={:.2f}'.format(*rgb_to_hsv_single(100, 100, 100))
#     # 'h=0.00 s=0.00 v=100.00'
#     # >>> rgb_to_hsv(np.array([[50, 120, 239], [100, 100, 100]]))
#     # array([[   0.6 ,    0.79,  239.  ],
#     #        [   0.  ,    0.  ,  100.  ]])
#     # """
#     input_shape = rgb.shape
#     rgb = rgb.reshape(-1, 3)
#     r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
#
#     maxc = np.maximum(np.maximum(r, g), b)
#     minc = np.minimum(np.minimum(r, g), b)
#     v = maxc
#
#     deltac = maxc - minc
#     s = deltac / maxc
#     deltac[deltac == 0] = 1  # to not divide by zero (those results in any way would be overridden in next lines)
#     rc = (maxc - r) / deltac
#     gc = (maxc - g) / deltac
#     bc = (maxc - b) / deltac
#
#     h = 4.0 + gc - rc
#     h[g == maxc] = 2.0 + rc[g == maxc] - bc[g == maxc]
#     h[r == maxc] = bc[g == maxc] - gc[g == maxc]
#     h[minc == maxc] = 0.0
#
#     h = (h / 6.0) % 1.0
#     res = np.dstack([h, s, v])
#     return res.reshape(input_shape)


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in xrange(len(images) / batch_size):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir),
        cifar_generator(['test_batch'], batch_size, data_dir)
)