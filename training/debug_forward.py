import os
import numpy as np
import mindspore as ms
from mindspore import context, Tensor

from tfnet_model import TFNetModel


def main():
    # Context
    try:
        context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    except Exception:
        context.set_context(mode=context.PYNATIVE_MODE)

    # Create dummy input: (B, T, C, H, W)
    B, T, C, H, W = 2, 8, 3, 112, 112
    x = np.random.randn(B, T, C, H, W).astype(np.float32)
    x = Tensor(x)

    # Lengths
    len_list = [8, 5]

    # Build model
    hidden_size = 256
    vocab_size = 100
    model = TFNetModel(hidden_size=hidden_size, word_set_num=vocab_size, device_target="CPU", dataset_name='CE-CSL')

    # Forward (eval mode path to be simpler)
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

