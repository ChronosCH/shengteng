import os, re, cv2
from natsort import natsorted  # 可选, 若未安装下面提供备用实现

# -------- 可配置项 ----------
image_folder = r"D:\Programs\shengteng\mind_vac\test\5"   # 修改为你的图片文件夹
output_video = r"D:\Programs\shengteng\mind_vac\test_vedios\output5.mp4"
fps = 30
# ----------------------------

# 如果你不想安装额外包，可以使用下面的自然排序 key 函数（已内建）
def natural_key(s):
    parts = re.findall(r'\d+|\D+', s)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key

# 列出 png/jpg 文件并自然排序
files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
if not files:
    raise SystemExit("没有找到图片，请检查 image_folder 路径和后缀")
files.sort(key=natural_key)  # 自然排序

# 读第一帧确定尺寸
first_path = os.path.join(image_folder, files[0])
first_img = cv2.imread(first_path)
if first_img is None:
    raise SystemExit(f"无法读取第一张图片: {first_path}")
h, w = first_img.shape[:2]

# 创建 VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 通用写 mp4
out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

for fname in files:
    path = os.path.join(image_folder, fname)
    img = cv2.imread(path)
    if img is None:
        print("跳过无法读取的文件：", path)
        continue
    # 若尺寸不一致，统一为第一个尺寸
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h))
    out.write(img)

out.release()
print("完成，输出文件：", os.path.abspath(output_video))
