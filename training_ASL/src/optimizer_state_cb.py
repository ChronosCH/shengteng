import os
from typing import Optional, Set

try:
    from mindspore.train.callback import Callback
    from mindspore import save_checkpoint
except Exception:
    class Callback:  # type: ignore
        def step_end(self, run_context):
            pass
    def save_checkpoint(obj, fpath):  # type: ignore
        raise RuntimeError("MindSpore not available")


class OptimizerStateCallback(Callback):
    """在模型 ckpt 保存后，保存匹配的优化器状态到同名 *_optim.ckpt，并同步 latest_optim.ckpt。"""

    def __init__(self, optimizer, alias_callback, checkpoint_dir: str):
        super().__init__()
        self.optimizer = optimizer
        self.alias_callback = alias_callback
        self.checkpoint_dir = checkpoint_dir
        self._paired_done: Set[str] = set()

    def _optim_path(self, model_ckpt: str) -> str:
        base, ext = os.path.splitext(model_ckpt)
        # 如果传入的就是 *_optim.ckpt（或被多次追加），去掉所有结尾的 _optim 再追加一次
        while base.endswith('_optim'):
            base = base[:-6]
        return base + "_optim.ckpt"

    def _copy_latest_optim(self, optim_ckpt: str):
        try:
            import shutil
            dst = os.path.join(self.checkpoint_dir, 'latest_optim.ckpt')
            if os.path.abspath(optim_ckpt) == os.path.abspath(dst):
                return
            shutil.copy2(optim_ckpt, dst)
        except Exception as e:
            print(f"[CKPT] 更新 latest_optim 失败: {e}")

    def step_end(self, run_context):
        if self.alias_callback is None:
            return
        try:
            model_ckpt: Optional[str] = self.alias_callback.get_last_ckpt()
        except Exception:
            model_ckpt = None
        if not model_ckpt or not os.path.exists(model_ckpt):
            return
        optim_ckpt = self._optim_path(model_ckpt)
        if optim_ckpt == model_ckpt:
            return
        if optim_ckpt in self._paired_done or os.path.exists(optim_ckpt):
            # 即便已存在，也尝试更新 latest_optim 别名
            if os.path.exists(optim_ckpt):
                self._copy_latest_optim(optim_ckpt)
            return
        try:
            save_checkpoint(self.optimizer, optim_ckpt)
            self._paired_done.add(optim_ckpt)
            print(f"[CKPT] 优化器状态已保存: {os.path.basename(optim_ckpt)}")
            self._copy_latest_optim(optim_ckpt)
        except Exception as e:
            print(f"[CKPT] 保存优化器状态失败: {e}")
