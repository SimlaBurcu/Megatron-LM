import torch
import torch.nn.functional as F
from torch.optim import SGD
import numpy as np
import pdb

class rounding_modes:
    STOC, DETERM = 'stoc', 'determ'
    modes = [STOC, DETERM]

def round_tensor(t, mode):
    #print("--- round_tensor ---")
    #pdb.set_trace()
    if mode == rounding_modes.STOC:
        sampled = torch.cuda.FloatTensor(t.size()).uniform_(-0.5, 0.5)
        return sampled.add_(t).round()
    elif mode == rounding_modes.DETERM:
        return t.round()
    else:
        raise NotImplementedError("Rounding mode %s is not implemented", mode)

def get_exp(t, epsilon):
    #print("--- get_exp ---")
    #pdb.set_trace()
    t = t.abs()
    max_v, _ = t.max(dim=1, keepdim=True)
    return (max_v + epsilon).log2().ceil()


def constrain_fp(t, mant_bits, epsilon):
    #print("--- constrain_fp ---")
    #pdb.set_trace()
    t_ = t.abs()
    exp = (t_+epsilon).log2().ceil()
    exp_mult = torch.pow(2.0, exp)

    t /= exp_mult

    mask = 2.0 ** float(mant_bits)

    t *= mask
    t = round_tensor(t, 'stoc')
    t /= mask

    t *= exp_mult

    return t

# TODO: make all of this in place.
def _float_to_bfp(t, mant_bits, epsilon, rounding_mode, exp_given=None):
    #print("--- _float_to_bfp ---")
    #print(mant_bits)
    #print(t.shape)
    exp = get_exp(t, epsilon)

    min_v = torch.pow(2.0, exp-mant_bits)
    max_v = torch.pow(2.0, exp) - min_v

    t = t/min_v

    rounded = round_tensor(t, rounding_mode)
    rounded *=  min_v

    return torch.min(torch.max(rounded, -max_v), max_v)


def float_to_bfp_batched(t, mant_bits, epsilon, rounding_mode, bfp_tile_size=25,
                         bfp_symmetric=False, num_format='', weight_mant_bits='',forward_step=False):
    #print("--- float_to_bfp_batched ---")
    #print(t.shape)
    #pdb.set_trace()
    assert num_format == 'bfp'
    if forward_step:
        mant_bits = 7

    orig_shape = t.size()

    t = t.view(t.size()[0], -1)
    o = _float_to_bfp(t, mant_bits, epsilon, rounding_mode)
    return o.view(orig_shape)


"""
"""

def get_tiled(t, orig_shape, bfp_tile_size):
    #print("--- get_tiled ---")
    #pdb.set_trace()
    t = t.view(orig_shape[0], -1)
    matrix_h, matrix_w = t.size()

    h_tiles = (matrix_h + bfp_tile_size - 1) // bfp_tile_size
    w_tiles = (matrix_w + bfp_tile_size - 1) // bfp_tile_size

    matrix_h_pad = h_tiles*bfp_tile_size
    matrix_w_pad = w_tiles*bfp_tile_size

    h_pad = matrix_h_pad - matrix_h
    w_pad = matrix_w_pad - matrix_w

    t = F.pad(t, (0, w_pad, 0, h_pad),'constant')
    # t <-h_tiles, tile_h, matrix_w
    t = t.view(h_tiles, bfp_tile_size, matrix_w_pad)
    # t <- h_tiles, matrix_w, tile_h,
    t.transpose_(1, 2)
    return (t.contiguous().view(h_tiles*w_tiles, -1),
            h_tiles, w_tiles,
            matrix_h, matrix_w,
            matrix_h_pad, matrix_w_pad)

def tiled_to_tensor(t, orig_shape, bfp_tile_size,
                    h_tiles, w_tiles,
                    matrix_h, matrix_w,
                    matrix_h_pad, matrix_w_pad):

    #print("--- tiled_to_tensor ---")
    #pdb.set_trace()
    # t <- h_tiles, w_tiles, tile_w, tile_h
    t = t.view(h_tiles, w_tiles, bfp_tile_size, bfp_tile_size)
    # t <- h_tiles, w_tiles, tile_h, tile_w
    t.transpose_(2, 3)
    # t <- h_tiles, tile_h, w_tiles, tile_w
    t.transpose_(1, 2)
    t = t.contiguous().view(matrix_h_pad, matrix_w_pad)
    return t.narrow(0, 0, matrix_h).narrow(1, 0, matrix_w).view(orig_shape)


def float_to_bfp_tiled(t, mant_bits, epsilon, rounding_mode, bfp_tile_size=25,
                       bfp_symmetric=False, num_format='', weight_mant_bits=0,
                       sgd_update=False, mant_bits_pow=None):
    #print("--- float_to_bfp_tiled ---")
    #print(t.shape)
    #pdb.set_trace()
    assert num_format == 'bfp'
    if sgd_update:
        mant_bits = weight_mant_bits

    orig_shape = t.size()
    if bfp_tile_size == 0:
        if bfp_symmetric:
            t = t.view(1, -1)
            t_avg = t.mean(dim=1, keepdim=True)
            return (t_avg + _float_to_bfp(t-t_avg, mant_bits, epsilon,
                                          rounding_mode)).view(orig_shape)
        else:
            return _float_to_bfp(t.view(1, -1), mant_bits, epsilon,
                                 rounding_mode).view(orig_shape)

    (t, h_tiles, w_tiles,
     matrix_h, matrix_w,
     matrix_h_pad, matrix_w_pad) = get_tiled(t, orig_shape, bfp_tile_size)

    if bfp_symmetric:
        t_avg = t.mean(dim=1, keepdim=True)
        t = t_avg + _float_to_bfp(t-t_avg, mant_bits, epsilon, rounding_mode)
    else:
        t = _float_to_bfp(t, mant_bits, epsilon, rounding_mode)

    return tiled_to_tensor(t, orig_shape, bfp_tile_size,
                           h_tiles, w_tiles,
                           matrix_h, matrix_w,
                           matrix_h_pad, matrix_w_pad)

def extract_weights(t, mant_bits, epsilon, rounding_mode, bfp_tile_size=25,
                    bfp_symmetric=False, num_format='', weight_mant_bits=0,
                    sgd_update=False, mant_bits_pow=None):
    #print("--- extract_weights ---")
    #pdb.set_trace()
    if weight_mant_bits == mant_bits:
        return t
    else:
        return constrain_fp(t, mant_bits, epsilon)


def _get_op_name(name, epsilon, mant_bits, rounding_mode, **kwargs):
    return  '%s_BFP_%s_%d' % (name, rounding_mode, mant_bits)

def _gen_bfp_op(op, name, bfp_args):
    #print("--- _gen_bfp_op ---")
    """
    Do the 'sandwitch'
    With an original op:

    out = op(x, y)
    grad_x, grad_y = op_grad(grad_out)

    To the following:
    x_, y_ = input_op(x, y)
    Where input_op(x, y) -> bfp(x), bfp(y)
    and input_op_grad(grad_x, grad_y) -> bfp(grad_x), bfp(grad_y)

    out_ = op(x_, y_)

    out = output_op(out)
    Where output_op(out) -> bfp(out)
    and output_op_grad(grad_out) -> bfp(grad_out)

    This way we garantee that everything in and out of the forward and backward operations is
    properly converted to bfp
    """
    # https://discuss.pytorch.org/t/call-backward-on-function-inside-a-backpropagation-step/3793/7

    #pdb.set_trace()
    name = _get_op_name(name, **bfp_args)

    class NewOpIn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            #print("--- NewOpIn forward ---")
            #print(bfp_args['mant_bits'])
            #bfp_args['mant_bits']=7
            return (float_to_bfp_batched(x, **bfp_args, forward_step=True),
                    w)

        @staticmethod
        def backward(ctx, grad_x, grad_w):
            #print("--- NewOpIn backward ---")
            #print(bfp_args['mant_bits'])
            return (grad_x, grad_w)

    NewOpIn.__name__ = name + '_In'
    new_op_in = NewOpIn.apply

    class NewOpOut(torch.autograd.Function):
        @staticmethod
        def forward(ctx, op_out):
            #print("--- NewOpOut forward ---")
            #print(bfp_args['mant_bits'])
            return op_out

        @staticmethod
        def backward(ctx, op_out_grad):
            #print("--- NewOpOut backward ---")
            #print(bfp_args['mant_bits'])
            bfp_args['mant_bits']=3
            return float_to_bfp_batched(op_out_grad, **bfp_args)

    NewOpOut.__name__ = name + '_Out'
    new_op_out = NewOpOut.apply

    def new_op(x, w, *args, **kwargs):
        #print("--- new_op ---")
        x, w = new_op_in(x, w)
        out = op(x, w, *args, **kwargs)
        return new_op_out(out)

    return new_op


_bfp_ops = {}


def _get_bfp_op(op, name, bfp_args):
    #print("--- _get_bfp_op ---")
    #pdb.set_trace()
    op_name = _get_op_name(name, **bfp_args)
    if op_name not in _bfp_ops:
        _bfp_ops[name] = _gen_bfp_op(op, name, bfp_args)

    return _bfp_ops[name]


def unpack_bfp_args(kwargs):
    #print("--- unpack_bfp_args ---")
    #pdb.set_trace()
    bfp_args = {}
    bfp_argn = [('num_format', 'fp32'),
                ('rounding_mode', 'stoc'),
                ('epsilon', 1e-8),
                ('mant_bits', 0),
                ('bfp_tile_size', 0),
                ('bfp_symmetric', False),
                ('weight_mant_bits', 0)]

    for arg, default in bfp_argn:
        if arg in kwargs:
            bfp_args[arg] = kwargs[arg]
            del kwargs[arg]
        else:
            bfp_args[arg] = default
    return bfp_args


def F_linear_bfp(**kwargs):
    #print("--- F_linear_bfp ---")
    #pdb.set_trace()
    bfp_args = unpack_bfp_args(kwargs)
    if bfp_args['num_format'] == 'bfp':
        return _get_bfp_op(F.linear, 'linear', bfp_args)
    else:
        return F.linear


class BFPConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, **kwargs):
        #print("--- BFPConv2d init ---")
        self.bfp_args = unpack_bfp_args(kwargs)

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.num_format = self.bfp_args['num_format']
        self.conv_op = _get_bfp_op(F.conv2d, 'Conv2d', self.bfp_args)

    def forward(self, input):
        #print("--- BFPConv2d forward ---")
        #pdb.set_trace()
        if self.num_format == 'fp32':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        elif self.num_format == 'bfp':
            conv = self.conv_op(input, self.weight, None, self.stride,
                                self.padding, self.dilation, self.groups)
            if self.bias is not None:
                return conv + self.bias
            else:
                return conv

        else:
            raise NotImplementedError('NumFormat not implemented')

class BFPLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        #print("--- BFPLinear init ---")
        self.bfp_args = unpack_bfp_args(kwargs)
        super().__init__(in_features, out_features, bias)
        self.num_format = self.bfp_args['num_format']
        self.linear_op = _get_bfp_op(F.linear, 'linear', self.bfp_args)

    def forward(self, input):
        #print("--- BFPLinear forward ---")
        #pdb.set_trace()
        if self.num_format == 'fp32':
            return F.linear(input, self.weight, self.bias)
        elif self.num_format == 'bfp':
            l = self.linear_op(input, self.weight, None)
            if self.bias is not None:
                return l + self.bias
            else:
                return l

        else:
            raise NotImplementedError('NumFormat not implemented')
