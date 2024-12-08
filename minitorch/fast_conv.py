from typing import Tuple

from numba import njit, prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function
import numpy as np

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # Process each element in output in parallel
    for element in prange(out_size):
        out_index : Index = np.zeros(3, np.int32)
        in_index : Index = np.zeros(3, np.int32)
        weights_index : Index = np.zeros(3, np.int32)

        to_index(element, out_shape, out_index)
        cur_batch, cur_out_channel, cur_width = out_index

        total = 0.0
        # Loop through each channel in input
        for curr_in_channel in range(in_channels):
            # Loop through each input element within the sliding window
            for position_in_window in range(kw):
                if not reverse:

                    # Padding
                    if cur_width + position_in_window >= width:
                        total += 0
                    else:
                        # Select corresponding weights and input value
                        weights_index[0], weights_index[1], weights_index[2] = cur_out_channel, curr_in_channel, position_in_window
                        in_index[0], in_index[1], in_index[2] = cur_batch, curr_in_channel, cur_width + position_in_window
                        total += input[index_to_position(in_index, s1)] * weight[index_to_position(weights_index, s2)]
                else:
                    # Padding
                    if cur_width - position_in_window < 0:
                        total += 0
                    else:
                        weights_index[0], weights_index[1], weights_index[2] = cur_out_channel, curr_in_channel, position_in_window
                        in_index[0], in_index[1], in_index[2] = cur_batch, curr_in_channel, cur_width - position_in_window
                        total += input[index_to_position(in_index, s1)] * weight[index_to_position(weights_index, s2)]

        out[index_to_position(out_index, out_strides)] = total


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
            batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    for element in prange(out_size):

        out_index : Index = np.zeros(4, np.int32)
        in_index : Index = np.zeros(4, np.int32)
        weights_index : Index = np.zeros(4, np.int32)
        to_index(element, out_shape, out_index)

        cur_batch, cur_out_channel, cur_h, cur_w = out_index
        total = 0
        for cur_in_channel in range(in_channels):
            for height_pos_offset in range(kh):
                for width_pos_offset in range(kw):
                    if not reverse:

                        if cur_h + height_pos_offset >= height or cur_w + width_pos_offset >= width:
                            total += 0
                        else:
                            weights_index[0], weights_index[1], weights_index[2], weights_index[3] = cur_out_channel, cur_in_channel, height_pos_offset, width_pos_offset
                            in_index[0], in_index[1], in_index[2], in_index[3] = cur_batch, cur_in_channel, cur_h + height_pos_offset, cur_w + width_pos_offset
                            total += input[index_to_position(in_index, s1)] * weight[index_to_position(weights_index, s2)]
                    else:

                        if cur_h - height_pos_offset < 0 or cur_w - width_pos_offset < 0:
                            total += 0
                        else:
                            weights_index[0], weights_index[1], weights_index[2], weights_index[3] = cur_out_channel, cur_in_channel, height_pos_offset, width_pos_offset
                            in_index[0], in_index[1], in_index[2], in_index[3] = cur_batch, cur_in_channel, cur_h - height_pos_offset, cur_w - width_pos_offset
                            total += input[index_to_position(in_index, s1)] * weight[index_to_position(weights_index, s2)]

        out[index_to_position(out_index, out_strides)] = total


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
