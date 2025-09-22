#!/usr/bin/env python3
import sys
import csv
sys.path.append('/home/cx857322378/shengteng/training')
from data_processor import preprocess_words

# 测试一行数据的处理
with open('/home/cx857322378/shengteng/training/data/CE-CSL/label/dev.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)
    
    # 处理第二行（第一行是表头）
    row = rows[1]
    print('原始行:')
    for i, field in enumerate(row):
        print(f'  Field {i}: "{field}"')
    
    print()
    print('处理后:')
    video_name = row[0].strip()
    translator = row[1].strip()
    gloss_text = row[3].strip()
    words = gloss_text.split('/')
    words = preprocess_words(words)
    
    print(f'Video name: "{video_name}"')
    print(f'Translator: "{translator}"')
    print(f'Gloss text: "{gloss_text}"')
    print(f'Words: {words[:10]}...')  # 只显示前10个词
    
    # 检查构建的路径
    video_path = f"data/CE-CSL/video/dev/{translator}/{video_name}.mp4"
    print(f'Video path: "{video_path}"')
