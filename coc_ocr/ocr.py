from paddlex import create_model
import cv2
import os
# ====================== 配置区 ======================
DET_MODEL_PATH = r"D:\PaddleX-release-3.7\output\det\best_accuracy\inference"
REC_MODEL_PATH = r"D:\PaddleX-release-3.7\output\rec\best_accuracy\inference"
TEST_IMAGE = r"D:\coc_bat\dataset\img_crop\fish_21.png"
SAVE_CROP_DIR = r"D:\coc_bat\dataset\crop_text"
# ====================================================

os.makedirs(SAVE_CROP_DIR, exist_ok=True)

det_model = create_model(
    model_name="PP-OCRv5_server_det",
    model_dir=DET_MODEL_PATH,
    device="gpu"
)
rec_model = create_model(
    model_name="PP-OCRv5_server_rec",
    model_dir=REC_MODEL_PATH,
    device="gpu"
)
# 检测推理
det_result = det_model.predict(TEST_IMAGE)
for res in det_result:
    res.save_to_img(r"D:\coc_bat\dataset\out_box_det")    # 自动保存【画检测框图片】
    dt_polys = res["dt_polys"]
    dt_scores = res["dt_scores"]

img = cv2.imread(TEST_IMAGE)
text_result_list = []
for idx, poly in enumerate(dt_polys):
    # poly = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    xmin = int(min(xs))
    ymin = int(min(ys))
    xmax = int(max(xs))
    ymax = int(max(ys))

    # 裁剪文字区域
    crop = img[ymin:ymax, xmin:xmax]
    crop_save_path = os.path.join(SAVE_CROP_DIR, f"crop_{idx}.jpg")
    cv2.imwrite(crop_save_path, crop)

    # 识别
    rec_out = list(rec_model.predict(crop_save_path))[0]
    rec_out.save_to_img(crop_save_path)  # 控制台打印结果
    text = rec_out["rec_text"].strip()
    score = rec_out["rec_score"]
    det_score = dt_scores[idx]

    text_result_list.append({
        "box": [xmin, ymin, xmax, ymax],
        "text": text,
        "rec_score": score,
        "det_score": det_score
    })
    print(f"📦框{idx} | 文本:{text} | 检测置信度:{det_score:.3f} | 识别置信度:{score:.3f}")

print("\n=======全部识别结果汇总=======")
for item in text_result_list:
    print(item)