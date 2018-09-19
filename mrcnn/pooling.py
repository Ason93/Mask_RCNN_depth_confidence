# #!usr/bin/env python
# # -*- coding:utf-8 -*-
# import numpy as np
# def pooling(inputMap, poolSize=3, poolStride=2, mode='max'):
#     """INPUTS:
#               inputMap - input array of the pooling layer
#               poolSize - X-size(equivalent to Y-size) of receptive field
#               poolStride - the stride size between successive pooling squares
#
#        OUTPUTS:
#                outputMap - output array of the pooling layer
#
#        Padding mode - 'edge'
#     """
#     # inputMap sizes
#     in_row, in_col = np.shape(inputMap)
#
#     # outputMap sizes
#     out_row, out_col = int(np.floor(in_row / poolStride)), int(np.floor(in_col / poolStride))
#     row_remainder, col_remainder = np.mod(in_row, poolStride), np.mod(in_col, poolStride)
#     if row_remainder != 0:
#         out_row += 1
#     if col_remainder != 0:
#         out_col += 1
#     outputMap = np.zeros((out_row, out_col))
#
#     # padding
#     temp_map = np.lib.pad(inputMap, ((0, poolSize - row_remainder), (0, poolSize - col_remainder)), 'edge')
#
#     # max pooling
#     for r_idx in range(0, out_row):
#         for c_idx in range(0, out_col):
#             startX = c_idx * poolStride
#             startY = r_idx * poolStride
#             poolField = temp_map[startY:startY + poolSize, startX:startX + poolSize]
#             poolOut = np.max(poolField)
#             outputMap[r_idx, c_idx] = poolOut
#
#     # retrun outputMap
#     return outputMap
#
#
# # 测试实例
#
# test = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# test_result = pooling(test, 2, 2, 'max')
# print(test_result)

# # !/usr/bin/env python
# # -*- coding: utf-8 -*-
# import tensorflow as tf
# import numpy as np
#
# input = tf.constant(np.random.rand(3, 4))
# k = 2
# output = tf.nn.top_k(input, k).indices
# with tf.Session() as sess:
#     print(sess.run(input))
#     print(sess.run(output))

# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import keras.backend as K
import tensorflow as tf

a = K.variable(np.array([[[1, 2, 3],[4, 5, 6]],
                         [[-1, -2, -3],[-4, -5, -6]]]))
b = K.variable(np.array([[[3, 2, 1], [6, 5, 4]],
                         [[-3, -2, -1], [-6, -5, -4]]]))
c1 = K.concatenate([a, b], axis=0)
c2 = K.concatenate([a, b], axis=-1)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(c1))
    print(sess.run(c2))
    print('c1.shape:', c1.shape)
    print('c2.shape:', c2.shape)
