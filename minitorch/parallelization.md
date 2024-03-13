MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
d:\vscodeworkspace\minitorch\minitorch\fast_ops.py (161)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, d:\vscodeworkspace\minitorch\minitorch\fast_ops.py (161)
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        in_storage: Storage,                                                 |
        in_shape: Shape,                                                     |
        in_strides: Strides,                                                 |
    ) -> None:                                                               |
        aligned = (                                                          |
            len(out_strides) == len(in_strides)                              |
            and len(in_strides) == len(out_strides)                          |
            and np.all(out_strides == in_strides)----------------------------| #0
            and np.all(out_shape == in_shape)--------------------------------| #1
        )                                                                    |
        if aligned:                                                          |
            for out_pos in prange(out.size):---------------------------------| #2
                out[out_pos] = fn(in_storage[out_pos])                       |
        else:                                                                |
            for out_pos in prange(out.size):---------------------------------| #3
                out_index = np.empty_like(out_shape, dtype=np.int32)         |
                in_index = np.empty_like(in_shape, dtype=np.int32)           |
                to_index_by_strides(out_pos, out_strides, out_index)         |
                broadcast_index(out_index, out_shape, in_shape, in_index)    |
                in_pos = index_to_position(in_index, in_strides)             |
                out[out_pos] = fn(in_storage[in_pos])                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None







ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
d:\vscodeworkspace\minitorch\minitorch\fast_ops.py (212)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, d:\vscodeworkspace\minitorch\minitorch\fast_ops.py (212)
-----------------------------------------------------------------------------|loop #ID
    def _zip(                                                                |
        out: Storage,                                                        |
        out_shape: Shape,                                                    |
        out_strides: Strides,                                                |
        a_storage: Storage,                                                  |
        a_shape: Shape,                                                      |
        a_strides: Strides,                                                  |
        b_storage: Storage,                                                  |
        b_shape: Shape,                                                      |
        b_strides: Strides,                                                  |
    ) -> None:                                                               |
        aligned = (                                                          |
            len(out_strides) == len(a_strides)                               |
            and len(out_strides) == len(a_strides)                           |
            and np.all(out_strides == a_strides)-----------------------------| #4
            and np.all(out_shape == a_shape)---------------------------------| #5
            and len(out_strides) == len(b_strides)                           |
            and len(out_strides) == len(b_strides)                           |
            and np.all(out_strides == b_strides)-----------------------------| #6
            and np.all(out_shape == b_shape)---------------------------------| #7
        )                                                                    |
        if aligned:                                                          |
            for out_pos in prange(out.size):---------------------------------| #8
                out[out_pos] = fn(a_storage[out_pos], b_storage[out_pos])    |
        else:                                                                |
            for out_pos in prange(out.size):---------------------------------| #9
                out_index = np.empty_like(out_shape, dtype=np.int32)         |
                a_index = np.empty_like(a_shape, dtype=np.int32)             |
                b_index = np.empty_like(b_shape, dtype=np.int32)             |
                to_index_by_strides(out_pos, out_strides, out_index)         |
                broadcast_index(out_index, out_shape, a_shape, a_index)      |
                broadcast_index(out_index, out_shape, b_shape, b_index)      |
                a_pos = index_to_position(a_index, a_strides)                |
                b_pos = index_to_position(b_index, b_strides)                |
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 6 parallel for-
loop(s) (originating from loops labelled: #4, #5, #6, #7, #8, #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None







REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
d:\vscodeworkspace\minitorch\minitorch\fast_ops.py (270)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, d:\vscodeworkspace\minitorch\minitorch\fast_ops.py (270)
---------------------------------------------------------------------|loop #ID
    def _reduce(                                                     |
        out: Storage,                                                |
        out_shape: Shape,                                            |
        out_strides: Strides,                                        |
        a_storage: Storage,                                          |
        a_shape: Shape,                                              |
        a_strides: Strides,                                          |
        reduce_dim: int,                                             |
    ) -> None:                                                       |
        for out_pos in prange(out.size):-----------------------------| #11
            out_index = np.empty_like(out_shape, dtype=np.int32)     |
            a_index = np.empty_like(out_shape, dtype=np.int32)       |
            to_index_by_strides(out_pos, out_strides, out_index)     |
            a_index[:] = out_index-----------------------------------| #10
            for reduce_dim_idx in range(a_shape[reduce_dim]):        |
                a_index[reduce_dim] = reduce_dim_idx                 |
                a_pos = index_to_position(a_index, a_strides)        |
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #11, #10).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--11 is a parallel loop
   +--10 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--10 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--10 (serial)



Parallel region 0 (loop #11) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#11).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None







MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
d:\vscodeworkspace\minitorch\minitorch\fast_ops.py (292)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, d:\vscodeworkspace\minitorch\minitorch\fast_ops.py (292)
-------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                       |
    out: Storage,                                                  |
    out_shape: Shape,                                              |
    out_strides: Strides,                                          |
    a_storage: Storage,                                            |
    a_shape: Shape,                                                |
    a_strides: Strides,                                            |
    b_storage: Storage,                                            |
    b_shape: Shape,                                                |
    b_strides: Strides,                                            |
) -> None:                                                         |
    """                                                            |
    NUMBA tensor matrix multiply function.                         |
                                                                   |
    Should work for any tensor shapes that broadcast as long as    |
                                                                   |
    ```                                                            |
    assert a_shape[-1] == b_shape[-2]                              |
    ```                                                            |
                                                                   |
    Optimizations:                                                 |
                                                                   |
    * Outer loop in parallel                                       |
    * No index buffers or function calls                           |
    * Inner loop should have no global writes, 1 multiply.         |
                                                                   |
                                                                   |
    Args:                                                          |
        out (Storage): storage for `out` tensor                    |
        out_shape (Shape): shape for `out` tensor                  |
        out_strides (Strides): strides for `out` tensor            |
        a_storage (Storage): storage for `a` tensor                |
        a_shape (Shape): shape for `a` tensor                      |
        a_strides (Strides): strides for `a` tensor                |
        b_storage (Storage): storage for `b` tensor                |
        b_shape (Shape): shape for `b` tensor                      |
        b_strides (Strides): strides for `b` tensor                |
                                                                   |
    Returns:                                                       |
        None : Fills in `out`                                      |
    """                                                            |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0         |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0         |
                                                                   |
    assert a_shape[-1] == b_shape[-2], 'dim does not match!'       |
                                                                   |
    for out_pos in prange(out.size):-------------------------------| #0
        out_index = np.empty_like(out_shape)                       |
        to_index_by_strides(out_pos, out_strides, out_index)       |
        a_index = np.empty_like(a_shape)                           |
        b_index = np.empty_like(b_shape)                           |
        broadcast_index(out_index, out_shape, a_shape, a_index)    |
        broadcast_index(out_index, out_shape, b_shape, b_index)    |
                                                                   |
        target_dim = a_shape[-1]                                   |
                                                                   |
        for inner_pos in range(target_dim):                        |
            a_index[-1] = inner_pos                                |
            b_index[-2] = inner_pos                                |
            a_pos = index_to_position(a_index, a_strides)          |
            b_pos = index_to_position(b_index, b_strides)          |
            out[out_pos] += a_storage[a_pos] * b_storage[b_pos]    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #0).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None