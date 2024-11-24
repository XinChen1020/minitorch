from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:

            # Only perform partial reduction for tensors with size exceeding 1024
            # (Important for large batch/sample size)
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            # Perform final reduction if necessary
            if a.shape[dim] > 1024:
                final_out_shape = list(a.shape)
                final_out_shape[dim] = 1
                final_out = a.zeros(tuple(final_out_shape))

                threadsperblock = min(1024, out_a.size)
                blockspergrid = 1
                f[blockspergrid, threadsperblock](  # type: ignore
                    *final_out.tuple(), final_out.size, *out_a.tuple(), dim, start
                )
                return final_out
            else:
                return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)

            out[index_to_position(out_index, out_strides)] = fn(in_storage[index_to_position(in_index, in_strides)])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            out[index_to_position(out_index, out_strides)] = fn(a_storage[index_to_position(a_index, a_strides)],
                                                                b_storage[index_to_position(b_index, b_strides)])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """

    # Note:
    # Since an warp contains 32 threads, I will not implement the
    # optimization which perform reduction while loading the data from global memory into shared memory
    # for this practice sum kernel.
    # However, it's possible to use half of the blockDIM to perform reduction
    # if we set a larger blockDIM

    # Important Assumption: The code only work for an even blockDim
    # blockDim should be multiple to 32 for most cases
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # Load data from global memory into shared memory
    if i < size :
        cache[pos] = float(a[i])
    else:
        cache[pos] = 0.0

    # Wait until all values are loaded properly
    cuda.syncthreads()

    if i < size:

        # Perform reduction from shared memory using sequential addressing
        # Sequential addressing help to avoids bank conflicts
        # (same memory bank get multiple requests at once from threads in a given warp,
        #  which bank to use is decided by address/bit_number % num_banks)
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride:
                cache[pos] += cache[pos + stride]
            cuda.syncthreads()
            stride //= 2

        # Write result to global memory
        if pos == 0:
            out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.

    Returns:
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        # Important: This function only do partial reducation if the elements size
        # in the reduce_dim exceeds blockDim (1024 in this case).

        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        cache[pos] = reduce_value

        # Ignore threads outside of the output tensor size
        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            dim = a_shape[reduce_dim]

            # Calculate where to collect load the data from
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos

            if out_index[reduce_dim] < dim:

                # Update Cache
                cache[pos] = a_storage[index_to_position(out_index, a_strides)]
                cuda.syncthreads()

                # Perform sequential reduction in shared memory
                stride = BLOCK_DIM // 2
                while stride > 0:
                    if pos < stride and pos + stride < dim:
                        cache[pos] = fn(cache[pos], cache[pos + stride])
                    cuda.syncthreads()
                    stride //= 2

            if pos == 0:
                out[index_to_position(out_index, out_strides)] = cache[0]

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """
    This is a practice square MM kernel to prepare for matmul.


    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square
    """
    BLOCK_DIM = 32
    a_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    col = cuda.threadIdx.x
    row = cuda.threadIdx.y

    if row >= size or col >= size:
        return

    a_cache[row, col] = a[row * size + col]
    b_cache[row, col] = b[row * size + col]
    cuda.syncthreads()

    result = 0
    for k in range(size):
        result += a_cache[row, k] * b_cache[k, col]

    out[row * size + col] = result


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # Ensure broadcastable for batch shape
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Modified:
    #   Replacing original:
    #   The final position c[i, j]
    #   i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    #   j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    #   (This map each thread from its own location on the grid to the transpose location
    #     of the output tensor, rather than a 1:1 mapping)
    # The final position c[row, col]
    # Row-wise has the advantage of global memory coalescing and
    # provides 1:1 mapping to the output tensor, which is more convenient
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    local_col = cuda.threadIdx.x
    local_row = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[row, col]

    result = 0.0
    # Loop through tiles of A and B along the shared dimension
    for tile_offset in range(0, a_shape[2], BLOCK_DIM):
        # Compute global indices for the current tile
        tile_col = tile_offset + local_col  # Global column index for A
        tile_row = tile_offset + local_row  # Global row index for B

        # Load A's tile into shared memory
        if row < a_shape[1] and tile_col < a_shape[2]:
            a_shared[local_row, local_col] = a_storage[
                batch * a_batch_stride + row * a_strides[1] + tile_col * a_strides[2]
            ]

        # Load B's tile as its transpose into shared memory
        # so that we can access shared memory row-wise rather than column-wise
        if tile_row < b_shape[1] and col < b_shape[2]:
            b_shared[local_col, local_row] = b_storage[
                batch * b_batch_stride + tile_row * b_strides[1] + col * b_strides[2]
            ]

        # Synchronize threads to ensure all data is loaded
        cuda.syncthreads()

        # Compute the dot product using transposed B
        for idx in range(BLOCK_DIM):
            if idx + tile_offset < a_shape[2] and idx + tile_offset < b_shape[1]:
                result += a_shared[local_row, idx] * b_shared[local_col, idx]

        # Synchronize threads before loading the next tile
        cuda.syncthreads()

    # Write the final result to the output matrix
    if row < out_shape[1] and col < out_shape[2]:
        out[out_strides[0] * batch + out_strides[1] * row + out_strides[2] * col] = result


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
