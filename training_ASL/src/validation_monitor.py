# ASL手语识别验证监控系统
from __future__ import annotations

import os
import time
import json
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️ NumPy未安装，使用内置函数")
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ Matplotlib未安装，跳过图表生成")
    plt = None
    font_manager = None
    MATPLOTLIB_AVAILABLE = False
from datetime import datetime
from collections import defaultdict
from typing import Optional

try:
    import mindspore as ms
    from mindspore import nn, Tensor
    from mindspore.train.callback import Callback
    from mindspore.nn.metrics import Accuracy, Top5CategoricalAccuracy
    import mindspore.ops as ops
    MINDSPORE_AVAILABLE = True
except ImportError:
    print("⚠️ MindSpore未安装，使用模拟模式")
    MINDSPORE_AVAILABLE = False
    
    # 模拟MindSpore的Callback类
    class Callback:
        def step_end(self, run_context):
            pass

# 新增：尝试使用 sklearn 生成分类报告与混淆矩阵
try:
    from sklearn.metrics import confusion_matrix, classification_report
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


    def _configure_matplotlib_fonts():
        """为中文显示配置合适的 Matplotlib 字体。"""
        if not MATPLOTLIB_AVAILABLE or plt is None:
            return

        try:
            preferred_fonts = [
                "Noto Sans CJK SC",
                "Noto Sans CJK",
                "Source Han Sans SC",
                "Source Han Sans",
                "SimHei",
                "Microsoft YaHei",
                "WenQuanYi Micro Hei",
                "Arial Unicode MS",
            ]

            available_fonts = {font.name for font in font_manager.fontManager.ttflist}
            chosen_font = None

            for font_name in preferred_fonts:
                if font_name in available_fonts:
                    chosen_font = font_name
                    break

            if chosen_font is None:
                fallback_files = [
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/noto/NotoSansCJKsc-Regular.otf",
                    "/usr/share/fonts/truetype/noto/NotoSansCJK.ttc",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/arphic/ukai.ttc",
                ]

                for font_path in fallback_files:
                    if os.path.isfile(font_path):
                        try:
                            font_manager.fontManager.addfont(font_path)
                            chosen_font = font_manager.FontProperties(fname=font_path).get_name()
                            break
                        except Exception:
                            continue

            if chosen_font:
                plt.rcParams['font.family'] = chosen_font
            else:
                plt.rcParams['font.sans-serif'] = preferred_fonts + list(plt.rcParams.get('font.sans-serif', []))
                print("⚠️ 未找到可用的中文字体，可能仍会出现缺失警告。建议安装 Noto Sans CJK。")

            plt.rcParams['axes.unicode_minus'] = False
        except Exception as err:
            print(f"⚠️ Matplotlib 中文字体配置失败: {err}")


    _configure_matplotlib_fonts()


class ValidationMonitor(Callback):
    """详细的验证监控器（集成评估与早停、混淆矩阵/报告落盘）"""
    
    def __init__(self, model, val_dataset, eval_interval=1, save_dir="results", early_stopping_patience: int = 10):
        super(ValidationMonitor, self).__init__()
        self.model = model
        self.val_dataset = val_dataset
        self.eval_interval = eval_interval
        self.save_dir = save_dir
        self.patience = early_stopping_patience if (early_stopping_patience and early_stopping_patience > 0) else None
        self.no_improve_epochs = 0
        
        # 创建结果保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 记录历史
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_top5_accuracy': [],
            'learning_rate': [],
            'timestamp': [],
            'top_k': [],
        }
        
        # 最佳结果记录
        self.best_accuracy = 0.0
        self.best_epoch = 0
        
        # 分类准确率统计
        self.class_accuracies = defaultdict(list)
        
        print(f"📊 验证监控器初始化完成")
        print(f"📁 结果保存至: {save_dir}")
        
        self._alias_cb = None  # 可选：CheckpointAliasCallback
    
    def attach_alias_callback(self, alias_cb):
        self._alias_cb = alias_cb

    def _find_latest_ckpt(self, ckpt_dir: str) -> str | None:
        try:
            files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
                     if f.endswith('.ckpt') and (not f.endswith('_optim.ckpt'))]
            if not files:
                return None
            files.sort(key=lambda p: os.path.getmtime(p))
            return files[-1]
        except Exception:
            return None

    def _copy_as(self, src: str, dst_name: str):
        import shutil
        try:
            dst = os.path.join(os.path.dirname(src), dst_name)
            if os.path.abspath(src) == os.path.abspath(dst):
                return
            shutil.copy2(src, dst)
            print(f"[CKPT] 更新 {dst_name} -> {os.path.basename(src)}")
        except Exception as e:
            print(f"[CKPT] 更新 {dst_name} 失败: {e}")

    def epoch_end(self, run_context):
        """每个epoch结束时的验证与早停判断"""
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        
        if cur_epoch % self.eval_interval != 0:
            return
        print(f"\n{'='*60}")
        print(f"📈 第 {cur_epoch} 轮验证开始 [{datetime.now().strftime('%H:%M:%S')}]")
        print(f"{'='*60}")
        
        # 执行详细验证（包含混淆矩阵/报告落盘）
        val_results = self.detailed_evaluation()
        
        # 记录结果
        self.record_results(cur_epoch, val_results, cb_params)
        
        # 保存历史与绘图
        self.save_results()
        self.plot_progress()

        # 早停逻辑
        if val_results['accuracy'] > self.best_accuracy:
            self.no_improve_epochs = 0
            self.best_accuracy = val_results['accuracy']
            self.best_epoch = cur_epoch
            # 以 alias 回调的最近 ckpt 为源，否则找最新
            src_ckpt = None
            if self._alias_cb is not None:
                try:
                    src_ckpt = self._alias_cb.get_last_ckpt()
                except Exception:
                    src_ckpt = None
            if src_ckpt is None:
                checkpoints_dir = os.path.dirname(self.save_dir)
                src_ckpt = self._find_latest_ckpt(checkpoints_dir)
            if src_ckpt:
                self._copy_as(src_ckpt, 'best.ckpt')
                # 同步 best_optim
                try:
                    base, ext = os.path.splitext(src_ckpt)
                    optim_src = base + '_optim.ckpt'
                    if os.path.exists(optim_src):
                        self._copy_as(optim_src, 'best_optim.ckpt')
                except Exception as e:
                    print(f"[CKPT] 同步 best_optim 失败: {e}")
                # 保存 best 状态
                try:
                    history_file = os.path.join(self.save_dir, "training_history.json")
                    if os.path.exists(history_file):
                        with open(history_file, 'r', encoding='utf-8') as f:
                            history = json.load(f)
                        best_state = {
                            'epoch': self.best_epoch,
                            'best_accuracy': self.best_accuracy,
                            'history_tail': {k: v[-5:] for k, v in history.items() if isinstance(v, list)},
                            'best_model_ckpt': os.path.join(os.path.dirname(self.save_dir), 'best.ckpt'),
                            'best_optim_ckpt': os.path.join(os.path.dirname(self.save_dir), 'best_optim.ckpt'),
                        }
                        with open(os.path.join(os.path.dirname(self.save_dir), 'training_state_best.json'), 'w', encoding='utf-8') as f:
                            json.dump(best_state, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[CKPT] 保存 best 状态失败: {e}")
        else:
            self.no_improve_epochs += 1
            if self.patience is not None and self.no_improve_epochs >= self.patience:
                print(f"⛔ 早停触发：连续 {self.patience} 个验证周期未提升，停止训练。")
                run_context.request_stop()

        # 同步 latest.ckpt
        checkpoints_dir = os.path.dirname(self.save_dir)
        latest = self._find_latest_ckpt(checkpoints_dir)
        if latest:
            self._copy_as(latest, 'latest.ckpt')

        print(f"{'='*60}")
    
    # 新增：独立的详细评估，返回并额外生成混淆矩阵与分类报告
    def detailed_evaluation(self):
        """详细的验证评估"""
        start_time = time.time()
        
        # 验证前切换到 eval 模式（禁用 Dropout/使用BN的推理统计）
        model_was_training = None
        try:
            if hasattr(self.model, 'training'):
                model_was_training = bool(getattr(self.model, 'training'))
            if hasattr(self.model, 'set_train'):
                self.model.set_train(False)
        except Exception:
            pass
        
        try:
            # 初始化统计
            total_samples = 0
            correct_predictions = 0
            topk_correct = 0
            total_loss = 0.0
            
            # 分类别统计
            class_correct = defaultdict(int)
            class_total = defaultdict(int)
            
            # 预测结果存储
            all_predictions = []
            all_labels = []
            all_confidences = []
            
            print("🔄 正在进行验证...")
            
            batch_count = 0
            # 根据模型实际类别数动态选择 top-k，避免类别数少于 5 时触发 CUDA assert
            current_topk = 5
            if MINDSPORE_AVAILABLE:
                loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
                for data in self.val_dataset.create_dict_iterator():
                    batch_count += 1
                    if batch_count % 50 == 0:
                        print(f"   处理批次: {batch_count}")

                    frames = data['frames']
                    labels = data['label']
                    batch_size = labels.shape[0]

                    # 统一标签类型与范围
                    try:
                        labels32 = ops.cast(labels, ms.int32)
                    except Exception:
                        labels32 = labels
                    try:
                        n_cls = int(getattr(self.model, 'num_classes', 0))
                    except Exception:
                        n_cls = 0
                    if n_cls and n_cls > 0:
                        current_topk = max(1, min(5, n_cls))
                        try:
                            lbl_min = Tensor(0, ms.int32)
                            lbl_max = Tensor(n_cls - 1, ms.int32)
                            try:
                                labels32 = ops.clip_by_value(labels32, lbl_min, lbl_max)
                            except Exception:
                                labels32 = ops.minimum(ops.maximum(labels32, lbl_min), lbl_max)
                        except Exception:
                            pass

                    # 前向传播
                    logits = self.model(frames)
                    # 在 AMP/O2 下 logits 可能为 float16，某些算子不支持 FP16 -> 显式转为 FP32
                    try:
                        logits32 = ops.cast(logits, ms.float32)
                    except Exception:
                        logits32 = logits

                    # 安全检查：确保标签索引未越界
                    num_logits = None
                    try:
                        num_logits = int(logits32.shape[1])
                    except Exception:
                        pass
                    if num_logits is not None:
                        labels_np = None
                        try:
                            labels_np = labels32.asnumpy()
                        except Exception:
                            pass
                        if labels_np is not None:
                            min_label = int(labels_np.min()) if labels_np.size else None
                            max_label = int(labels_np.max()) if labels_np.size else None
                            if (max_label is not None and max_label >= num_logits) or (min_label is not None and min_label < 0):
                                print(f"[VAL][ERROR] 标签越界: min_label={min_label}, max_label={max_label}, logits_dim={num_logits}")
                                raise RuntimeError(f"Validation labels out of range (max={max_label}, logits_dim={num_logits})")

                    # 计算损失（使用 FP32 logits）
                    batch_loss = loss_fn(logits32, labels32)
                    try:
                        total_loss += float(batch_loss.asnumpy()) * int(batch_size)
                    except Exception:
                        total_loss += float(batch_loss) * int(batch_size)

                    # 计算预测（用 FP32 提高稳定性），并转为 NumPy 避免 GPU gather 断言
                    probabilities = nn.Softmax()(logits32)
                    try:
                        probs_np = probabilities.asnumpy()
                    except Exception:
                        probs_np = np.array(probabilities)
                    try:
                        labels_np = labels32.asnumpy()
                    except Exception:
                        labels_np = np.array(labels32)
                    if labels_np.dtype.kind != 'i':
                        labels_np = labels_np.astype(np.int32)

                    pred_np = probs_np.argmax(axis=1)
                    correct_batch = int((pred_np == labels_np).sum())
                    correct_predictions += correct_batch

                    # Top-5准确率（纯 NumPy 计算）
                    top_k_val = int(current_topk)
                    top_k_val = max(1, min(top_k_val, probs_np.shape[1]))
                    if top_k_val >= probs_np.shape[1]:
                        topk_indices = np.argsort(-probs_np, axis=1)
                        topk_indices = topk_indices[:, :top_k_val]
                    else:
                        topk_indices = np.argpartition(-probs_np, top_k_val - 1, axis=1)[:, :top_k_val]
                    batch_topk_hits = int(sum(1 for idx, label in enumerate(labels_np) if label in topk_indices[idx]))
                    topk_correct += batch_topk_hits

                    total_samples += int(batch_size)

                    # 记录详细结果
                    conf_np = probs_np.max(axis=1)

                    all_predictions.extend(pred_np.tolist() if hasattr(pred_np, 'tolist') else list(pred_np))
                    all_labels.extend(labels_np.tolist() if hasattr(labels_np, 'tolist') else list(labels_np))
                    all_confidences.extend(conf_np.tolist() if hasattr(conf_np, 'tolist') else list(conf_np))

                    # 分类别统计
                    for pred, true_label in zip(pred_np, labels_np):
                        class_total[int(true_label)] += 1
                        if int(pred) == int(true_label):
                            class_correct[int(true_label)] += 1
            else:
                # 模拟验证结果
                print("   使用模拟数据进行验证...")
                total_samples = 1000
                correct_predictions = int(total_samples * 0.75)  # 模拟75%准确率
                topk_correct = int(total_samples * 0.92)  # 模拟92% Top-5准确率
                total_loss = 0.8 * total_samples
                all_predictions = [0]*total_samples
                all_labels = [0]*total_samples
                all_confidences = [0.5]*total_samples
        
        finally:
            # 恢复训练模式
            try:
                if hasattr(self.model, 'set_train'):
                    self.model.set_train(True if model_was_training else False)
            except Exception:
                pass
        
        eval_time = time.time() - start_time
        
        # 计算最终指标
        val_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        val_top5_accuracy = topk_correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_confidence = float(np.mean(all_confidences)) if all_confidences else 0
        
        # 分类别准确率
        class_accuracies = {}
        for class_id in class_total:
            if class_total[class_id] > 0:
                class_accuracies[class_id] = class_correct[class_id] / class_total[class_id]
        
        # 生成并保存混淆矩阵与分类报告
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            if _HAS_SKLEARN and len(all_labels) > 0:
                cm = confusion_matrix(all_labels, all_predictions)
                cm_path = os.path.join(self.save_dir, 'confusion_matrix.npy')
                np.save(cm_path, cm)
                print(f"✅ 混淆矩阵已保存: {cm_path}")
                # PNG 可视化
                if MATPLOTLIB_AVAILABLE:
                    plt.figure(figsize=(8, 6))
                    plt.imshow(cm, interpolation='nearest', cmap='Blues')
                    plt.title('Confusion Matrix')
                    plt.colorbar()
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label')
                    plt.tight_layout()
                    cm_png = os.path.join(self.save_dir, 'confusion_matrix.png')
                    plt.savefig(cm_png, dpi=200)
                    plt.close()
                    print(f"✅ 混淆矩阵图已保存: {cm_png}")
                # 文本报告
                report = classification_report(all_labels, all_predictions, digits=4)
                rep_path = os.path.join(self.save_dir, 'classification_report.txt')
                with open(rep_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"✅ 分类报告已保存: {rep_path}")
        except Exception as e:
            print(f"⚠️ 保存混淆矩阵/报告失败: {e}")
        
        # 输出验证结果
        print(f"\n📊 验证结果:")
        print(f"   总样本数: {total_samples:,}")
        print(f"   验证损失: {avg_loss:.4f}")
        print(f"   验证准确率: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"   Top-{current_topk}准确率: {val_top5_accuracy:.4f} ({val_top5_accuracy*100:.2f}%)")
        print(f"   平均置信度: {avg_confidence:.4f}")
        print(f"   验证耗时: {eval_time:.2f}秒")
        
        # 显示最佳类别和最差类别
        if class_accuracies:
            best_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)[:3]
            worst_classes = sorted(class_accuracies.items(), key=lambda x: x[1])[:3]
            
            print(f"\n🏆 表现最佳的类别:")
            for class_id, acc in best_classes:
                print(f"   类别 {class_id}: {acc:.4f} ({acc*100:.2f}%)")
            
            print(f"\n⚠️ 需要改进的类别:")
            for class_id, acc in worst_classes:
                print(f"   类别 {class_id}: {acc:.4f} ({acc*100:.2f}%)")
        
        return {
            'accuracy': val_accuracy,
            'top5_accuracy': val_top5_accuracy,
            'loss': avg_loss,
            'confidence': avg_confidence,
            'eval_time': eval_time,
            'class_accuracies': class_accuracies,
            'total_samples': total_samples,
            'top_k': current_topk,
        }
    
    def record_results(self, epoch, val_results, cb_params):
        """记录验证结果"""
        # 获取训练损失
        train_loss = getattr(cb_params, 'net_outputs', 0.0)
        if hasattr(train_loss, 'asnumpy'):
            train_loss = float(train_loss.asnumpy())
        
        # 获取学习率
        learning_rate = 1e-4
        try:
            lr = getattr(cb_params, 'optimizer', None)
            if lr and hasattr(lr, 'get_lr'):
                lr_val = lr.get_lr()
                if hasattr(lr_val, 'asnumpy'):
                    learning_rate = float(lr_val.asnumpy())
                else:
                    learning_rate = float(lr_val)
        except Exception:
            # 获取失败则使用默认
            pass
        
        # 添加到历史记录
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_results['loss'])
        self.history['val_accuracy'].append(val_results['accuracy'])
        self.history['val_top5_accuracy'].append(val_results['top5_accuracy'])
        self.history['learning_rate'].append(learning_rate)
        self.history['timestamp'].append(datetime.now().isoformat())
        self.history['top_k'].append(val_results.get('top_k', 5))
        
        # 更新最佳结果
        if val_results['accuracy'] > self.best_accuracy:
            self.best_accuracy = val_results['accuracy']
            self.best_epoch = epoch
            print(f"\n🎉 新的最佳准确率! {self.best_accuracy:.4f} (轮次 {epoch})")
        
        # 更新分类准确率历史
        for class_id, acc in val_results['class_accuracies'].items():
            self.class_accuracies[class_id].append(acc)
    
    def save_results(self):
        """保存验证结果"""
        # 保存历史记录
        history_file = os.path.join(self.save_dir, "training_history.json")
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        
        # 保存最佳结果摘要
        summary = {
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.history['epoch']),
            'final_accuracy': self.history['val_accuracy'][-1] if self.history['val_accuracy'] else 0,
            'final_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0,
            'last_updated': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(self.save_dir, "best_results.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def plot_progress(self):
        """绘制训练进度图表"""
        if len(self.history['epoch']) < 2:
            return
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'ASL训练进度 - 最佳准确率: {self.best_accuracy:.4f}', fontsize=16)
        
        epochs = self.history['epoch']
        
        # 损失曲线
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='训练损失', alpha=0.7)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='验证损失')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('训练/验证损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot(epochs, [acc*100 for acc in self.history['val_accuracy']], 'g-', label='Top-1准确率')
        axes[0, 1].plot(epochs, [acc*100 for acc in self.history['val_top5_accuracy']], 'orange', label='Top-k准确率')
        axes[0, 1].axhline(y=self.best_accuracy*100, color='r', linestyle='--', alpha=0.7, label=f'最佳: {self.best_accuracy*100:.2f}%')
        axes[0, 1].set_xlabel('轮次')
        axes[0, 1].set_ylabel('准确率 (%)')
        axes[0, 1].set_title('验证准确率')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率曲线
        axes[1, 0].plot(epochs, self.history['learning_rate'], 'purple', label='学习率')
        axes[1, 0].set_xlabel('轮次')
        axes[1, 0].set_ylabel('学习率')
        axes[1, 0].set_title('学习率变化')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 准确率提升趋势
        if len(self.history['val_accuracy']) > 1:
            accuracy_diff = np.diff(self.history['val_accuracy'])
            axes[1, 1].bar(epochs[1:], accuracy_diff, alpha=0.7, 
                          color=['green' if x > 0 else 'red' for x in accuracy_diff])
            axes[1, 1].set_xlabel('轮次')
            axes[1, 1].set_ylabel('准确率变化')
            axes[1, 1].set_title('每轮准确率变化')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_file = os.path.join(self.save_dir, "training_progress.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 进度图表已保存: {plot_file}")
        
        plt.close()
    
    def get_summary(self):
        """获取训练摘要"""
        if not self.history['epoch']:
            return "暂无训练记录"
        
        current_epoch = self.history['epoch'][-1]
        current_acc = self.history['val_accuracy'][-1]
        current_loss = self.history['val_loss'][-1]
        
        summary = f"""
📊 ASL训练进度摘要
{'='*50}
📈 当前轮次: {current_epoch}
🎯 当前准确率: {current_acc:.4f} ({current_acc*100:.2f}%)
📉 当前损失: {current_loss:.4f}
🏆 最佳准确率: {self.best_accuracy:.4f} ({self.best_accuracy*100:.2f}%)
🎖️ 最佳轮次: {self.best_epoch}
⏱️ 总训练时长: {len(self.history['epoch'])} 轮次
{'='*50}
        """
        
        return summary


def create_validation_report(history_file="results/training_history.json"):
    """创建详细的验证报告"""
    if not os.path.exists(history_file):
        print(f"❌ 找不到历史文件: {history_file}")
        return
    
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    if not history['epoch']:
        print("❌ 没有训练历史记录")
        return
    
    print("📋 生成详细验证报告...")
    
    # 基本统计
    epochs = history['epoch']
    accuracies = history['val_accuracy']
    losses = history['val_loss']
    
    best_acc_idx = np.argmax(accuracies)
    worst_acc_idx = np.argmin(accuracies)
    
    # 生成报告
    report = f"""
📊 ASL手语识别训练验证报告
{'='*80}

📈 基本统计:
   总训练轮次: {len(epochs)}
   最终准确率: {accuracies[-1]:.4f} ({accuracies[-1]*100:.2f}%)
   最终损失: {losses[-1]:.4f}
   
🏆 最佳表现:
   最佳准确率: {accuracies[best_acc_idx]:.4f} ({accuracies[best_acc_idx]*100:.2f}%)
   最佳轮次: {epochs[best_acc_idx]}
   
📉 表现分析:
   准确率范围: {min(accuracies):.4f} - {max(accuracies):.4f}
   准确率标准差: {np.std(accuracies):.4f}
   平均准确率: {np.mean(accuracies):.4f}
   
🔄 收敛分析:
   准确率提升次数: {sum(1 for i in range(1, len(accuracies)) if accuracies[i] > accuracies[i-1])}
   准确率下降次数: {sum(1 for i in range(1, len(accuracies)) if accuracies[i] < accuracies[i-1])}
   最大单次提升: {max(accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))):.4f}
   
⏱️ 时间信息:
   训练开始: {history['timestamp'][0]}
   最后更新: {history['timestamp'][-1]}
   
{'='*80}
    """
    
    print(report)
    
    # 保存报告
    report_file = os.path.join(os.path.dirname(history_file), "validation_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📁 详细报告已保存: {report_file}")


if __name__ == "__main__":
    # 测试验证监控器
    print("🧪 测试验证监控器...")
    
    # 创建测试监控器
    monitor = ValidationMonitor(None, None, save_dir="results/test")
    
    # 模拟训练历史
    for epoch in range(1, 11):
        # 模拟验证结果
        accuracy = 0.6 + 0.03 * epoch + np.random.normal(0, 0.01)
        accuracy = min(max(accuracy, 0), 1)  # 限制在0-1范围
        
        val_results = {
            'accuracy': accuracy,
            'top5_accuracy': accuracy + 0.1,
            'loss': 1.0 - accuracy * 0.8,
            'confidence': 0.7 + accuracy * 0.2,
            'eval_time': 30.0,
            'class_accuracies': {i: accuracy + np.random.normal(0, 0.05) for i in range(10)},
            'total_samples': 1000
        }
        
        # 记录结果
        class MockParams:
            net_outputs = val_results['loss']
        
        monitor.record_results(epoch, val_results, MockParams())
    
    # 保存和可视化
    monitor.save_results()
    monitor.plot_progress()
    
    # 输出摘要
    print(monitor.get_summary())
    
    # 生成报告
    create_validation_report(os.path.join(monitor.save_dir, "training_history.json"))
