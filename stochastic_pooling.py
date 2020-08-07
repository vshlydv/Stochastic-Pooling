import numpy
from theano.compat.six.moves import xrange
import theano
from theano import tensor
from theano.gof.op import get_debug_values
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import contains_inf

pool_size = None
stride_size = None

# Compute index in pooled space of last needed pool
def last_pool(im_shp, p_shp, p_strd):
    rval = int(numpy.ceil(float(im_shp - p_shp) / p_strd))
    return rval
    
def stochastic_max_pool_x(x, image_shape, pool_shape = (2, 2), pool_stride = (1, 1), rng = None):
    """
    Parameters
    ----------
    x : theano 4-tensor
        in format (batch size, channels, rows, cols)
    image_shape : tuple
        avoid doing some of the arithmetic in theano
    pool_shape : tuple
        shape of the pool region (rows, cols)
    pool_stride : tuple
        strides between pooling regions (row stride, col stride)
    rng : theano random stream
    """

    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride
    global pool_size
    pool_size = pool_shape
    global stride_size
    stride_size = pool_stride
    batch = x.shape[0]
    channel = x.shape[1]
    rng = make_theano_rng(rng, 2022, which_method='multinomial')
    
    # Compute starting row of the last pool
    last_pool_r = last_pool(r, pr, rs) * rs
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(c, pc, cs) * cs
    required_c = last_pool_c + pc

    # final result shape
    res_r = int(numpy.floor(last_pool_r/rs)) + 1
    res_c = int(numpy.floor(last_pool_c/cs)) + 1

    # padding
    padded = tensor.alloc(0.0, batch, channel, required_r, required_c) 
    #theano.tensor.alloc(value, *shape) - for allocating a new tensor with value filled with "value"

    x = tensor.set_subtensor(padded[:,:, 0:r, 0:c], x) 
    #theano.tensor.set_subtensor(lval of = operator, rval of = operator) - for assigning a tensor to a subtensor of a tensor
    

    # unraveling    
    window = tensor.alloc(0.0, batch, channel, res_r, res_c, pr, pc)

    # initializing window with proper values
    for row_within_pool in xrange(pr):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pc):
            col_stop = last_pool_c + col_within_pool + 1
            win_cell = x[:,:,row_within_pool:row_stop:rs, col_within_pool:col_stop:cs]
            window  =  tensor.set_subtensor(window[:,:,:,:, row_within_pool, col_within_pool], win_cell)

    # find the norm
    norm = window.sum(axis = [4, 5])
    #tensor.sum(axis = []) - cal sum over given axes

    norm = tensor.switch(tensor.eq(norm, 0.0), 1.0, norm)
    '''
    theano.tensor.eq(a, b) - Returns a variable representing the result of logical equality (a==b)
    theano.tensor.switch(cond, ift, iff) - Returns a variable representing a switch between ift (iftrue) and iff (iffalse)
    Basically converting a zero norm to 1.0.
    '''
    norm = window / norm.dimshuffle(0, 1, 2, 3, 'x', 'x')
    #converting activation values to probabilities using below formula - pi = ai / sum(ai)

    # get prob
    prob = rng.multinomial(pvals = norm.reshape((batch * channel * res_r * res_c, pr * pc)), dtype='float32')
    # select
    res = (window * prob.reshape((batch, channel, res_r, res_c,  pr, pc))).max(axis=5).max(axis=4)

    return tensor.cast(res, theano.config.floatX)
    #tensor.cast() - for type casting the value of the tensor


def output_shape_of_lambda(input_shape):
    x = (input_shape[0], input_shape[1], int(numpy.floor((input_shape[2] - pool_size[0]) / stride_size[0] + 1)), int(numpy.floor((input_shape[3] - pool_size[1]) / stride_size[1] + 1)))
    return x

