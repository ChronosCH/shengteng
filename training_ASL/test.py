import os
import sys
print("=== 环境验证 ===")
print("Python版本:", sys.version.split()[0])
print("Python路径:", sys.executable)

# 检查LD_LIBRARY_PATH
ld_path = os.environ.get("LD_LIBRARY_PATH", "")
print("LD_LIBRARY_PATH前120字符:", (ld_path[:120] + "...") if ld_path else "(空)")

# 检查基础包
try:
    import numpy as np
    print("✅ NumPy版本:", np.__version__)
except ImportError as e:
    print("❌ NumPy导入失败:", e)

try:
    import cv2
    print("✅ OpenCV版本:", cv2.__version__)
except ImportError as e:
    print("❌ OpenCV导入失败:", e)

# 检查MindSpore
try:
    import mindspore as ms
    from mindspore import Tensor, ops
    print("✅ MindSpore版本:", ms.__version__)
    
    # 尝试GPU上下文
    try:
        ms.set_context(device_target="GPU")
        x = Tensor(np.ones((2,3), dtype=np.float32))
        y = Tensor(np.ones((2,3), dtype=np.float32))
        out = ops.add(x, y)
        print("✅ GPU加法测试成功，结果和=", float(out.asnumpy().sum()))
        print("✅ 当前device_target =", ms.get_context("device_target"))
    except Exception as gpu_e:
        print("⚠️  GPU上下文初始化失败:", repr(gpu_e))
        print("   尝试CPU模式...")
        try:
            ms.set_context(device_target="CPU")
            x = Tensor(np.ones((2,3), dtype=np.float32))
            y = Tensor(np.ones((2,3), dtype=np.float32))
            out = ops.add(x, y)
            print("✅ CPU加法测试成功，结果和=", float(out.asnumpy().sum()))
            print("✅ 当前device_target =", ms.get_context("device_target"))
        except Exception as cpu_e:
            print("❌ CPU模式也失败:", repr(cpu_e))
            
except ImportError as e:
    print("❌ MindSpore导入失败:", e)

print("=== 验证完成 ===")
