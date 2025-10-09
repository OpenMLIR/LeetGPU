import cutlass.cute as cute
import cutlass

@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,  # coordinate tensor
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # slice for CTAs
    # logical id -> address
    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]  # (TileM,TileN)
    blkB = gB[blk_coord]  # (TileM,TileN)
    blkC = gC[blk_coord]  # (TileM,TileN)
    blkCrd = cC[blk_coord]  # (TileM, TileN)

    # Note: these prints only run at compile/jit time
    print(f"[DSL INFO] Sliced Tensors per thread block:")
    print(f"[DSL INFO]   blkA = {blkA.type}")
    print(f"[DSL INFO]   blkB = {blkB.type}")
    print(f"[DSL INFO]   blkC = {blkC.type}")
    print(f"[DSL INFO]   blkCrd = {blkCrd.type}")

    # # declare the atoms which will be used later for memory copy
    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_C = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)

    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)

    # allocate fragments for gmem->rmem
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    thrCrd = thr_copy_C.partition_S(blkCrd)
    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)

    print(f"[DSL INFO] Sliced Tensors per thread:")
    print(f"[DSL INFO]   thrA = {thrA.type}")
    print(f"[DSL INFO]   thrB = {thrB.type}")
    print(f"[DSL INFO]   thrC = {thrC.type}")
    print(f"[DSL INFO]   thrCrd = {thrCrd.type}")

    for i in range(0, cute.size(frgPred), 1):
        val = cute.elem_less(thrCrd[i], shape)
        frgPred[i] = val

    # Print per thread predicate mask
    # if tidx == 0 and bidx == 0:
    #     cute.printf("block_dim = {}", cute.arch.grid_dim())
    #     cute.printf("shape = {}", shape)
    #     cute.print_tensor(thrA)
    #     cute.print_tensor(thrB)
    #     cute.print_tensor(frgPred)

    ##########################################################
    # Move data to reg address space
    ##########################################################

    cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom_load, thrB, frgB, pred=frgPred)

    # if tidx == 0 and bidx == 0:
    #     cute.print_tensor(frgA)
    #     cute.print_tensor(frgB)

    # Load data before use. The compiler will optimize the copy and load
    # operations to convert some memory ld/st into register uses.
    result = frgA.load() + frgB.load()

    # Save the results back to registers. Here we reuse b's registers.
    frgC.store(result)

    # Copy the results back to c
    cute.copy(copy_atom_store, frgC, thrC, pred=frgPred)


@cute.jit
def elementwise_add(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):

    dtype = mA.element_type
    vector_size = copy_bits // dtype.width

    thr_layout = cute.make_ordered_layout((1, 128), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, vector_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"[DSL INFO] Input Tensors:")
    print(f"[DSL INFO]   mA = {mA.type}")
    print(f"[DSL INFO]   mB = {mB.type}")

    print(f"[DSL INFO] Tiling Parameters:")
    print(f"[DSL INFO]   tiler_mn = {tiler_mn} per thread block")
    print(f"[DSL INFO]   tv_layout = {tv_layout}")

    gA = cute.zipped_divide(mA, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    gB = cute.zipped_divide(mB, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    gC = cute.zipped_divide(mC, tiler_mn)  # ((TileM,TileN),(RestM,RestN))
    print(f"[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA.type}")
    print(f"[DSL INFO]   gB = {gB.type}")
    print(f"[DSL INFO]   gC = {gC.type}")

    idC = cute.make_identity_tensor(mC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)
    print(f"[DSL INFO]   coord tensor = {cC.type}")

    elementwise_add_kernel(gA, gB, gC, cC, mC.shape, thr_layout, val_layout).launch(
        grid=[cute.size(gC, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )

@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    layout = cute.make_layout((1, N), stride=(N, 1))
    A = cute.make_tensor(A.iterator, layout)
    B = cute.make_tensor(B.iterator, layout)
    C = cute.make_tensor(C.iterator, layout)
    elementwise_add(A, B, C, 512)
