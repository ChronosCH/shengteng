import os
import numpy as np
import mindspore as ms
from mindspore import context, nn, Tensor
from mindspore.train.callback import Callback
import pandas as pd
from config import Config

from .model import ASLRecognitionModel
from .data_loader import ASLDataset, VideoProcessor, get_class_mapping


class ASLPredictor:
    """ASL手语识别预测器"""
    
    def __init__(self, model_path, config):
        self.config = config
        
        # 设置MindSpore环境
        context.set_context(
            mode=context.GRAPH_MODE,
            device_target="GPU" if config.get('use_gpu', True) else "CPU",
            device_id=config.get('device_id', 0)
        )
        
        # 加载类别映射
        train_csv = os.path.join(config['data_dir'], "splits", "train.csv")
        self.class_to_idx, self.idx_to_class = get_class_mapping(train_csv)
        self.num_classes = len(self.class_to_idx)
        
        # 创建模型
        self.model = ASLRecognitionModel(
            num_classes=self.num_classes,
            sequence_length=config['sequence_length'],
            input_size=config['input_size'],
            base_channels=config.get('base_channels', 64)
        )
        
        # 加载权重
        if os.path.exists(model_path):
            param_dict = ms.load_checkpoint(model_path)
            ms.load_param_into_net(self.model, param_dict)
            print(f"已加载模型权重: {model_path}")
        else:
            print(f"警告: 模型文件不存在 {model_path}")
        
        self.model.set_train(False)
        
        # 创建视频处理器
        self.processor = VideoProcessor(
            target_size=config['input_size'],
            sequence_length=config['sequence_length']
        )
    
    def predict_single_video(self, video_path):
        """预测单个视频"""
        # 提取帧
        frames = self.processor.extract_frames(video_path)
        
        if frames is None:
            return None, 0.0
        
        # 转换维度和增加批次维度
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C, T, H, W)
        frames = np.expand_dims(frames, axis=0)  # (1, C, T, H, W)
        
        # 转换为Tensor
        frames_tensor = Tensor(frames, ms.float32)
        
        # 预测
        logits = self.model(frames_tensor)
        probabilities = ms.ops.softmax(logits, axis=1)
        
        # 获取预测结果
        predicted_idx = ms.ops.argmax(probabilities, axis=1).asnumpy()[0]
        confidence = probabilities[0, predicted_idx].asnumpy()
        predicted_class = self.idx_to_class[predicted_idx]
        
        return predicted_class, float(confidence)
    
    def predict_batch(self, video_paths):
        """批量预测"""
        results = []
        
        for video_path in video_paths:
            if not os.path.exists(video_path):
                results.append((None, 0.0))
                continue
            
            predicted_class, confidence = self.predict_single_video(video_path)
            results.append((predicted_class, confidence))
        
        return results
    
    def evaluate_predictions(self, csv_path, video_dir):
        """评估预测结果"""
        # 读取测试数据
        df = pd.read_csv(csv_path)
        
        correct = 0
        total = 0
        predictions = []
        
        print("正在进行预测评估...")
        
        for idx, row in df.iterrows():
            video_file = row['Video file']
            true_class = row['Gloss']
            
            video_path = os.path.join(video_dir, video_file)
            
            if not os.path.exists(video_path):
                continue
            
            predicted_class, confidence = self.predict_single_video(video_path)
            
            predictions.append({
                'video_file': video_file,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'correct': predicted_class == true_class if predicted_class else False
            })
            
            if predicted_class == true_class:
                correct += 1
            total += 1
            
            if total % 100 == 0:
                print(f"已处理 {total} 个样本, 当前准确率: {correct/total:.4f}")
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n评估完成!")
        print(f"总样本数: {total}")
        print(f"正确预测: {correct}")
        print(f"准确率: {accuracy:.4f}")
        
        return predictions, accuracy


class ASLDemo:
    """ASL演示类"""
    
    def __init__(self, model_path, config):
        self.predictor = ASLPredictor(model_path, config)
        self.config = config
    
    def demo_random_samples(self, num_samples=10):
        """随机选择样本进行演示"""
        test_csv = os.path.join(self.config['data_dir'], "splits", "test.csv")
        video_dir = os.path.join(self.config['data_dir'], "videos")
        
        # 读取测试数据
        df = pd.read_csv(test_csv)
        
        # 随机选择样本
        random_samples = df.sample(n=min(num_samples, len(df)))
        
        print(f"\n=== 随机演示 {len(random_samples)} 个样本 ===")
        
        for idx, row in random_samples.iterrows():
            video_file = row['Video file']
            true_class = row['Gloss']
            
            video_path = os.path.join(video_dir, video_file)
            
            if not os.path.exists(video_path):
                print(f"视频文件不存在: {video_file}")
                continue
            
            predicted_class, confidence = self.predictor.predict_single_video(video_path)
            
            status = "✓" if predicted_class == true_class else "✗"
            
            print(f"\n{status} 视频: {video_file}")
            print(f"   真实标签: {true_class}")
            print(f"   预测标签: {predicted_class}")
            print(f"   置信度: {confidence:.4f}")
    
    def run_interactive_demo(self):
        """交互式演示"""
        print("\n=== ASL手语识别交互式演示 ===")
        print("输入视频文件名 (相对于videos目录) 或 'quit' 退出")
        
        video_dir = os.path.join(self.config['data_dir'], "videos")
        
        while True:
            video_name = input("\n请输入视频文件名: ").strip()
            
            if video_name.lower() == 'quit':
                break
            
            video_path = os.path.join(video_dir, video_name)
            
            if not os.path.exists(video_path):
                print(f"视频文件不存在: {video_path}")
                continue
            
            print("正在预测...")
            predicted_class, confidence = self.predictor.predict_single_video(video_path)
            
            print(f"预测结果: {predicted_class}")
            print(f"置信度: {confidence:.4f}")


def main():
    """主函数"""
    
    # 配置
    config = Config.get_train_config('medium')
    
    # 模型路径 (根据实际训练结果调整)
    # 尝试查找最新模型
    ckpt_dir = config['checkpoint_dir']
    model_path = None
    if os.path.exists(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        if ckpts:
            model_path = os.path.join(ckpt_dir, sorted(ckpts)[-1])
    if model_path is None:
        print("未找到模型检查点，请先训练模型。")
        return
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型文件")
        return
    
    # 创建预测器
    predictor = ASLPredictor(model_path, config)
    
    # 评估测试集
    test_csv = os.path.join(config['data_dir'], "splits", "test.csv")
    video_dir = os.path.join(config['data_dir'], "videos")
    
    predictions, accuracy = predictor.evaluate_predictions(test_csv, video_dir)
    
    # 保存预测结果
    results_df = pd.DataFrame(predictions)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(Config.RESULTS_DIR, "test_predictions.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"预测结果已保存到: {out_csv}")
    
    # 运行演示
    demo = ASLDemo(model_path, config)
    demo.demo_random_samples(10)
    
    # 交互式演示
    demo.run_interactive_demo()


if __name__ == "__main__":
    main()
