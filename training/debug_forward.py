import os
import numpy as np
import mindspore as ms
from mindspore import context, Tensor

from tfnet_model import TFNetModel


def main():
    # 设置上下文
    try:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    except Exception:
        context.set_context(mode=context.PYNATIVE_MODE)

    # 创建虚拟输入数据: (B, T, C, H, W)
    B, T, C, H, W = 2, 8, 3, 112, 112
    x = np.random.randn(B, T, C, H, W).astype(np.float32)
    x = Tensor(x)

    # 序列长度
    len_list = [8, 5]

    # 构建模型
    hidden_size = 256
    vocab_size = 100
    model = TFNetModel(hidden_size=hidden_size, word_set_num=vocab_size, device_target="CPU", dataset_name='CE-CSL')

    # 前向传播（评估模式路径更简单）
    model.set_train(False)
    outputs = model(x, len_list, is_train=False)
    log_probs1, log_probs2, log_probs3, log_probs4, log_probs5, lgt, *_ = outputs

    print("Shapes:")
    print("log_probs1:", tuple(log_probs1.shape))
    print("log_probs2:", tuple(log_probs2.shape))
    print("log_probs3:", tuple(log_probs3.shape))
    print("log_probs4:", tuple(log_probs4.shape))
    print("log_probs5:", tuple(log_probs5.shape))
    print("lgt:", lgt.asnumpy())


if __name__ == "__main__":
    main()

