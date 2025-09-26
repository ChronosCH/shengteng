import os
import shutil
from typing import Optional, List

try:
    from mindspore.train.callback import Callback
except Exception:
    # 兼容未安装 MindSpore 的静态分析环境
    class Callback:  # type: ignore
        def step_end(self, run_context):
            pass


class CheckpointAliasCallback(Callback):
    """在每次按步保存检查点后，同步生成 latest.ckpt，并提供最近一次保存的 ckpt 路径。
    - 依赖与 ModelCheckpoint 使用相同的 save_interval_steps 配置。
    - 通过比较目录中文件集合差异来定位新增的 ckpt 文件。
    """

    def __init__(self, checkpoint_dir: str, save_interval_steps: int, latest_name: str = 'latest.ckpt'):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_interval_steps = max(1, int(save_interval_steps))
        self.latest_name = latest_name
        self.last_saved_ckpt: Optional[str] = None
        self._known = set(self._list_ckpts())

    def _list_ckpts(self) -> List[str]:
        if not os.path.isdir(self.checkpoint_dir):
            return []
        items = []
        for f in os.listdir(self.checkpoint_dir):
            if not f.endswith('.ckpt') or f.endswith('_optim.ckpt'):
                continue
            if f == self.latest_name:
                continue
            items.append(os.path.join(self.checkpoint_dir, f))
        return items

    def get_last_ckpt(self) -> Optional[str]:
        if self.last_saved_ckpt and os.path.exists(self.last_saved_ckpt):
            return self.last_saved_ckpt
        # 回退：选取目录下最新修改时间的 ckpt（排除 *_optim.ckpt）
        files = self._list_ckpts()
        if not files:
            return None
        return max(files, key=lambda p: os.path.getmtime(p))

    def _copy_as_latest(self, src: str):
        try:
            dst = os.path.join(self.checkpoint_dir, self.latest_name)
            if os.path.abspath(src) == os.path.abspath(dst):
                return
            shutil.copy2(src, dst)
            print(f"[CKPT] latest.ckpt -> {os.path.basename(src)}")
        except Exception as e:
            print(f"[CKPT] 更新 latest 失败: {e}")

    def step_end(self, run_context):
        cb = run_context.original_args()
        cur_step = getattr(cb, 'cur_step_num', None)
        if cur_step is None:
            return
        if (cur_step % self.save_interval_steps) != 0:
            return
        files = self._list_ckpts()
        new_files = [f for f in files if f not in self._known]
        picked = None
        if new_files:
            picked = max(new_files, key=lambda p: os.path.getmtime(p))
        elif files:
            picked = max(files, key=lambda p: os.path.getmtime(p))
        if picked:
            self.last_saved_ckpt = picked
            self._known.update(files)
            self._copy_as_latest(picked)
