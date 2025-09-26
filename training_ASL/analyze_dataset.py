import os
from collections import Counter

try:
	import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas 不可用时走无依赖分支
	pd = None  # type: ignore

from config import Config


def _read_split(path: str):
	"""读取数据分割，优先使用 pandas，失败时回退到 csv.DictReader。"""
	if pd is not None:
		return pd.read_csv(path)  # type: ignore[no-any-return]

	import csv

	rows = []
	with open(path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for row in reader:
			rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
	return rows


def _count_samples(split):
	return len(split)


def _unique_glosses(split):
	if pd is not None:
		return set(split['Gloss'].astype(str).unique())
	return {str(row.get('Gloss', '')).strip() for row in split if row.get('Gloss')}


def _top_gloss_counts(split, top_n=10):
	if pd is not None:
		return split['Gloss'].value_counts().head(top_n)
	counter = Counter(str(row.get('Gloss', '')).strip() for row in split if row.get('Gloss'))
	return counter.most_common(top_n)


def main():
	# 分析数据集
	data_dir = Config.DATA_DIR
	splits_dir = os.path.join(data_dir, "splits")

	# 读取数据分割
	train_split = _read_split(os.path.join(splits_dir, "train.csv"))
	val_split = _read_split(os.path.join(splits_dir, "val.csv"))
	test_split = _read_split(os.path.join(splits_dir, "test.csv"))

	# 数据集统计
	train_count = _count_samples(train_split)
	val_count = _count_samples(val_split)
	test_count = _count_samples(test_split)

	print("数据集统计:")
	print(f"训练集样本数: {train_count}")
	print(f"验证集样本数: {val_count}")
	print(f"测试集样本数: {test_count}")
	print(f"总样本数: {train_count + val_count + test_count}")

	# 统计类别数量
	train_glosses = _unique_glosses(train_split)
	val_glosses = _unique_glosses(val_split)
	test_glosses = _unique_glosses(test_split)
	all_glosses = train_glosses | val_glosses | test_glosses
	print(f"\n手语类别总数: {len(all_glosses)}")

	print(f"\n训练集中的类别数: {len(train_glosses)}")
	print(f"验证集中的类别数: {len(val_glosses)}")
	print(f"测试集中的类别数: {len(test_glosses)}")

	# 显示一些样例类别
	print(f"\n前20个手语类别: {sorted(list(all_glosses))[:20]}")

	# 检查视频文件
	videos_dir = os.path.join(data_dir, "videos")
	video_files = os.listdir(videos_dir) if os.path.isdir(videos_dir) else []
	print(f"\n视频文件总数: {len(video_files)}")

	# 检查数据分布
	print("\n各分割中类别分布（前10个）:")
	print("训练集:")
	top_train = _top_gloss_counts(train_split)
	if pd is not None:
		print(top_train)
	else:
		for gloss, count in top_train:
			print(f"  {gloss}: {count}")

	print("验证集:")
	top_val = _top_gloss_counts(val_split)
	if pd is not None:
		print(top_val)
	else:
		for gloss, count in top_val:
			print(f"  {gloss}: {count}")

	print("测试集:")
	top_test = _top_gloss_counts(test_split)
	if pd is not None:
		print(top_test)
	else:
		for gloss, count in top_test:
			print(f"  {gloss}: {count}")


if __name__ == "__main__":
	main()
