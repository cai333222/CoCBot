import os
from PIL import Image

# ===================== 配置区 =====================
IMG_FOLDER = r"./20260731_151902_img"
CROP_FOLDER = r"./img_crop"
x1, x2 = 1000, 1800
y1, y2 = 880, 1300
# ==================================================

def get_save_path(base_dir, original_filename):
    name_no_ext, ext = os.path.splitext(original_filename)
    save_path = os.path.join(base_dir, original_filename)
    index = 1
    # 如果文件已存在，循环追加编号
    while os.path.exists(save_path):
        save_path = os.path.join(base_dir, f"{name_no_ext}_{index}{ext}")
        index += 1
    return save_path

def crop_images():
    os.makedirs(CROP_FOLDER, exist_ok=True)

    for filename in os.listdir(IMG_FOLDER):
        if filename.lower().endswith(".png"):
            src_path = os.path.join(IMG_FOLDER, filename)
            dst_path = get_save_path(CROP_FOLDER, filename)

            try:
                img = Image.open(src_path)
                crop_img = img.crop((x1, y1, x2, y2))
                crop_img.save(dst_path)
                print(f"✅ 裁剪完成: {filename} → {os.path.basename(dst_path)}")
            except Exception as e:
                print(f"❌ 处理失败 {filename} : {str(e)}")

if __name__ == "__main__":
    crop_images()