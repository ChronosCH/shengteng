#!/usr/bin/env python3
"""
cuBLAS错误修复补丁
专门解决 CUBLAS_STATUS_INVALID_VALUE 错误（错误码7）
"""

import mindspore as ms
import mindspore.ops as ops
import numpy as np

# ===== 调试开关 =====
ENABLE_CUBLAS_DEBUG = True  # 置为 False 可关闭详细日志
MAX_DEBUG_PRINT = 20        # 最多打印前 N 次 matmul 调试
_debug_invocations = ms.mutable(0)


def _tensor_stats(t, name):
    """安全获取张量统计信息（仅用于调试，避免在图模式抛出异常）"""
    try:
        arr = t.asnumpy()
        return (f"{name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4e}, "
                f"max={arr.max():.4e}, mean={arr.mean():.4e}, has_nan={np.isnan(arr).any()}, has_inf={np.isinf(arr).any()}")
    except Exception as e:
        return f"{name}: <stats_failed {e}> shape={t.shape} dtype={t.dtype}"


def apply_cublas_fixes():
    """应用cuBLAS错误的修复方案"""
    # 仅在未设置GPU时才调整，避免与训练脚本重复修改冲突
    try:
        ctx_target = ms.context.get_context("device_target")
        if ctx_target != "GPU":
            ms.context.set_context(
                device_target="GPU",
                mode=ms.context.PYNATIVE_MODE,
                max_device_memory="6GB"
            )
    except Exception:
        pass

    import os
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    print("✓ Applied cuBLAS context fixes")


def validate_tensor_for_matmul(tensor, name="tensor"):
    """验证张量是否适合矩阵乘法"""
    if not isinstance(tensor, ms.Tensor):
        raise TypeError(f"{name} must be a MindSpore Tensor")
    if tensor.dtype not in [ms.float32, ms.float16]:
        print(f"Warning: {name} dtype {tensor.dtype} may cause cuBLAS issues")
    shape = tensor.shape
    for i, dim in enumerate(shape):
        if dim <= 0:
            raise ValueError(f"{name} has invalid dimension {i}: {dim}")
    if len(shape) >= 2:
        M, N = shape[-2], shape[-1]
        if M <= 0 or N <= 0:
            raise ValueError(f"{name} has invalid matrix dimensions: M={M}, N={N}")
        if M % 4 != 0 or N % 4 != 0:
            print(f"Info: {name} dimensions ({M}, {N}) not aligned to 4, may be suboptimal")
    return True


def safe_matmul(a, b, name="matmul"):
    """安全的矩阵乘法（兼容GRAPH_MODE，增加诊断日志）"""
    global _debug_invocations
    try:
        # ====== 基础验证 ======
        validate_tensor_for_matmul(a, f"{name}_input_a")
        validate_tensor_for_matmul(b, f"{name}_input_b")

        # ====== 统一 dtype ======
        if a.dtype != ms.float32:
            a = ops.cast(a, ms.float32)
        if b.dtype != ms.float32:
            b = ops.cast(b, ms.float32)

        a_shape = a.shape
        b_shape = b.shape
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError(f"Matrices must be at least 2D: a={a_shape}, b={b_shape}")
        if a_shape[-1] != b_shape[-2]:
            raise ValueError(f"Matrix dimension mismatch: a[..., {a_shape[-1]}] @ b[{b_shape[-2]}, ...]")

        M = int(a_shape[-2]); K = int(a_shape[-1]); N = int(b_shape[-1])
        if M <= 0 or K <= 0 or N <= 0:
            raise ValueError(f"Invalid matrix dimensions for cuBLAS: M={M}, K={K}, N={N}")

        # ====== 数值健康检查 ======
        debug_print = False
        if ENABLE_CUBLAS_DEBUG and _debug_invocations < MAX_DEBUG_PRINT:
            debug_print = True
            _debug_invocations += 1
        if debug_print:
            print(f"[safe_matmul] #{_debug_invocations} BEGIN {name}")
            print(_tensor_stats(a, f"{name}_A"))
            print(_tensor_stats(b, f"{name}_B"))

        # 检测 NaN / Inf -> 替换为 0 以防 cuBLAS 报错
        try:
            isnan = ops.isnan(a)
            isinf = ops.isinf(a)
            if ops.reduce_any(isnan) or ops.reduce_any(isinf):
                if debug_print: print(f"{name}: A has NaN/Inf -> zeroing")
                a = ops.select(isnan | isinf, ops.zeros_like(a), a)
            isnan_b = ops.isnan(b)
            isinf_b = ops.isinf(b)
            if ops.reduce_any(isnan_b) or ops.reduce_any(isinf_b):
                if debug_print: print(f"{name}: B has NaN/Inf -> zeroing")
                b = ops.select(isnan_b | isinf_b, ops.zeros_like(b), b)
        except Exception:
            pass

        # ====== 维度对齐计算 ======
        pad_M, pad_K, pad_N = M, K, N
        if M % 4 != 0: pad_M = ((M + 3) // 4) * 4
        if K % 4 != 0: pad_K = ((K + 3) // 4) * 4
        if N % 4 != 0: pad_N = ((N + 3) // 4) * 4
        need_padding = (pad_M, pad_K, pad_N) != (M, K, N)
        if need_padding and debug_print:
            print(f"{name}: padding ({M},{K},{N}) -> ({pad_M},{pad_K},{pad_N})")

        # ====== 执行填充（仅 concat，不用原地） ======
        if need_padding:
            if pad_M != M:
                rows_add = pad_M - M
                a = ops.concat((a, ops.zeros(a_shape[:-2] + (rows_add, K), a.dtype)), axis=len(a_shape)-2)
            if pad_K != K:
                k_add = pad_K - K
                a = ops.concat((a, ops.zeros(a.shape[:-1] + (k_add,), a.dtype)), axis=len(a.shape)-1)
                b = ops.concat((b, ops.zeros(b_shape[:-2] + (k_add, N), b.dtype)), axis=len(b_shape)-2)
            if pad_N != N:
                n_add = pad_N - N
                b = ops.concat((b, ops.zeros(b.shape[:-1] + (n_add,), b.dtype)), axis=len(b.shape)-1)
            if debug_print:
                print(f"{name}: padded shapes A={a.shape} B={b.shape}")

        # ====== 执行 matmul ======
        result = ops.matmul(a, b)
        if need_padding:
            result = result[..., :M, :N]

        if debug_print:
            print(_tensor_stats(result, f"{name}_Result"))
            print(f"[safe_matmul] #{_debug_invocations} END {name}\n")
        return result

    except Exception as e:
        print(f"MatMul failed for {name}: {e}")
        print(f"  A shape={a.shape if 'a' in locals() else 'unknown'} B shape={b.shape if 'b' in locals() else 'unknown'}")
        try:
            print(f"  Attempting CPU fallback for {name}...")
            original_device = ms.context.get_context("device_target")
            ms.context.set_context(device_target="CPU")
            cpu_result = ops.matmul(a, b)
            ms.context.set_context(device_target=original_device)
            print(f"  ✓ CPU fallback successful for {name}")
            return cpu_result
        except Exception as cpu_e:
            print(f"  CPU fallback also failed: {cpu_e}")
            raise e


def fix_batch_processing_shapes(batch_data, expected_batch_size=None):
    """修复批次处理中的形状问题"""
    if not isinstance(batch_data, ms.Tensor):
        batch_data = ms.Tensor(batch_data, ms.float32)
    shape = batch_data.shape
    if len(shape) == 0:
        raise ValueError("Empty tensor shape")
    batch_size = int(shape[0])
    if batch_size <= 0:
        raise ValueError(f"Invalid batch size: {batch_size}")
    if expected_batch_size is not None and batch_size != expected_batch_size:
        if batch_size > expected_batch_size:
            batch_data = batch_data[:expected_batch_size]
        else:
            pad_size = expected_batch_size - batch_size
            pad_shape = (pad_size,) + shape[1:]
            pad_tensor = ops.zeros(pad_shape, batch_data.dtype)
            batch_data = ops.concat([batch_data, pad_tensor], axis=0)
    new_shape = []
    for dim in batch_data.shape:
        dim_val = max(1, int(dim))
        if len(new_shape) >= 1:
            dim_val = ((dim_val + 3) // 4) * 4
        new_shape.append(dim_val)
    if tuple(new_shape) != batch_data.shape:
        total_elements = 1
        for dim in batch_data.shape:
            total_elements *= dim
        new_total = 1
        for dim in new_shape:
            new_total *= dim
        if new_total > total_elements:
            pad_elements = new_total - total_elements
            flat_data = ops.reshape(batch_data, (-1,))
            pad_data = ops.zeros((pad_elements,), batch_data.dtype)
            flat_data = ops.concat([flat_data, pad_data], axis=0)
            batch_data = ops.reshape(flat_data, new_shape)
        else:
            batch_data = ops.reshape(batch_data, new_shape)
    return batch_data


def create_safe_linear_layer(input_dim, output_dim, name="linear"):
    """创建安全的线性层，避免cuBLAS错误"""
    input_dim = max(1, int(input_dim))
    output_dim = max(1, int(output_dim))
    input_dim_aligned = ((input_dim + 3) // 4) * 4
    output_dim_aligned = ((output_dim + 3) // 4) * 4
    limit = np.sqrt(6.0 / (input_dim_aligned + output_dim_aligned))
    weight = np.random.uniform(-limit, limit, (input_dim_aligned, output_dim_aligned)).astype(np.float32)
    if input_dim_aligned != input_dim or output_dim_aligned != output_dim:
        weight = weight[:input_dim, :output_dim]
    weight_tensor = ms.Tensor(weight, ms.float32)
    print(f"✓ Created safe linear layer {name}: ({input_dim}, {output_dim})")
    return weight_tensor


def diagnose_cublas_error(error_msg):
    """诊断cuBLAS错误并提供解决方案"""
    solutions = []
    if "CUBLAS_STATUS_INVALID_VALUE" in error_msg or "error 7" in error_msg:
        solutions.extend([
            "1. 检查矩阵维度是否匹配",
            "2. 确保所有张量为float32类型",
            "3. 验证批次大小是否为正数",
            "4. 检查是否存在0维或负维度",
            "5. 确保内存对齐（建议维度为4的倍数）"
        ])
    if "MatMul" in error_msg:
        solutions.extend([
            "6. 使用safe_matmul函数替代直接的ops.matmul",
            "7. 在矩阵乘法前添加形状验证",
            "8. 确保输入张量连续存储（使用.contiguous()如果可用）"
        ])
    if "For 'MatMul'" in error_msg:
        solutions.extend([
            "9. 检查线性层的输入特征维度",
            "10. 验证LSTM输出维度与分类器输入维度匹配",
            "11. 确保批次处理中每个样本的特征维度一致"
        ])
    print("🔍 cuBLAS错误诊断:")
    print(f"错误信息: {error_msg}")
    print("\n💡 建议解决方案:")
    for solution in solutions:
        print(f"  {solution}")
    return solutions


if __name__ == "__main__":
    print("Testing cuBLAS fixes...")
    apply_cublas_fixes()
    try:
        a = ms.Tensor(np.random.randn(13, 64).astype(np.float32))
        b = ms.Tensor(np.random.randn(64, 3512).astype(np.float32))
        result = safe_matmul(a, b, "test_matmul")
        print(f"✓ Safe MatMul test passed: {result.shape}")
    except Exception as e:
        print(f"✗ Safe MatMul test failed: {e}")
    try:
        batch_data = ms.Tensor(np.random.randn(4, 10, 64).astype(np.float32))
        fixed_data = fix_batch_processing_shapes(batch_data)
        print(f"✓ Batch processing fix test passed: {fixed_data.shape}")
    except Exception as e:
        print(f"✗ Batch processing fix test failed: {e}")
    print("cuBLAS fixes test completed!")
