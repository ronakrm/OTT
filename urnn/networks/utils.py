# Utilities etc.
import tensorflow as tf

def serialize_to_file(file, losses):
    file=open(file, 'w')
    for l in losses:
        file.write("{0}\n".format(l))
    file.close()


def unpool_2d(pool, 
              ind, 
              stride=[1, 2, 2, 1], 
              scope='unpool_2d'):
    """Adds a 2D unpooling op.
    https://arxiv.org/abs/1505.04366
    Unpooling layer after max_pool_with_argmax.
       Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                          shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)
    return ret


def unpool_3d(pool, 
              ind, 
              stride=[1, 2, 2, 2, 1], 
              scope='unpool_3d'):
    """Adds a 3D unpooling op.
    https://arxiv.org/abs/1505.04366
    Unpooling layer after max_pool_with_argmax.
       Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]*stride[3], input_shape[4]]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                          shape=[input_shape[0], 1, 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2], set_input_shape[3] * stride[3], set_input_shape[4]]
        ret.set_shape(set_output_shape)
    return ret

