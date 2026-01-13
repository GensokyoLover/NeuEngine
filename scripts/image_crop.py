import os
from PIL import Image

def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)

    left   = (w - s) // 2
    top    = (h - s) // 2
    right  = left + s
    bottom = top + s

    return img.crop((left, top, right, bottom))


def process_dir(
    input_dir,
    output_dir,
    exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"),
):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(exts):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        try:
            with Image.open(in_path) as img:
                img = img.convert("RGB")  # 防止 alpha / palette
                cropped = center_crop_square(img)

                # ⬇️ 上下翻转（Vertical Flip）
                cropped = cropped.transpose(Image.FLIP_TOP_BOTTOM)

                cropped.save(out_path)

        except Exception as e:
            print(f"[WARN] Skip {fname}: {e}")



if __name__ == "__main__":
    input_dir = r"H:\Falcor\emissive/"    # 改成你的目录
    output_dir = r"H:\Falcor\emissive_crop/"  # 输出目录

    process_dir(input_dir, output_dir)
