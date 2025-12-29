import json
import os
import requests
from urllib.parse import urlparse
# ====== 你要筛选的 ID ======
TARGET_IDS = {4241}   # 例如你想下载这些类别的

# ====== 你的 JSON 数据 ======
with open(r"H:\Falcor\scripts\python\bginfo.json", "r") as f:
    data = json.load(f)
# ====== 创建存储文件夹 ======
SAVE_DIR = "downloaded_images"
os.makedirs(SAVE_DIR, exist_ok=True)
def get_filename_from_url(url):
    path = urlparse(url).path
    filename = os.path.basename(path)
    return filename if filename else "unknown.jpg"
# ====== 下载函数 ======
def download_image(url, filename):
    try:
        r = requests.get(url, timeout=10)
        
        if r.status_code == 200:
            with open(filename, "wb") as f:
                f.write(r.content)
            print("✔ Saved:", filename)
        else:
            print("✖ Failed:", url, "-- Status:", r.status_code)
    except Exception as e:
        print("⚠ Error downloading", url, e)

for item in data:
    if item["transProdCatId"] in TARGET_IDS:
        url = item["previewImg"]
        filename = get_filename_from_url(url)
        save_path = os.path.join(SAVE_DIR, filename)

        download_image(url, save_path)