import uiautomator2 as u2
import cv2
import numpy as np

try:
    import tkinter as tk
except Exception:
    tk = None

scale = 1.0  # 用于把“显示坐标”换算为“原图坐标”
window_name = "coc"
screen_w = None
screen_h = None

if tk is not None:
    # 只获取一次屏幕尺寸，避免每帧都创建 Tk
    _root = tk.Tk()
    _root.withdraw()
    screen_w = _root.winfo_screenwidth()
    screen_h = _root.winfo_screenheight()
    _root.destroy()

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 显示图可能被等比缩小，所以这里把显示坐标换算回原始截图坐标
        orig_x = int(round(x / scale))
        orig_y = int(round(y / scale))
        print(f"✅ 点击坐标(原图): {orig_x}, {orig_y}  |  (显示): X={x}, Y={y}")

d = u2.connect()
print("进入实时截图模式...")
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, on_mouse_click)

while True:
    img = d.screenshot()

    # PIL 转 OpenCV 格式
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # 获取图片尺寸
    h, w = img_cv.shape[:2]

    # 根据屏幕尺寸决定是否缩小显示（避免窗口被裁剪），同时保证点击坐标可换算回原图
    scale = 1.0
    if screen_w is not None and screen_h is not None:
        # 预留一点边距，减少窗口仍超出屏幕的情况
        max_w = max(1, screen_w - 20)
        max_h = max(1, screen_h - 80)
        scale = min(1.0, max_w / w, max_h / h)

    if scale < 1.0:
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img_show = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_show = img_cv

    cv2.imshow(window_name, img_show)

    # 等待一小段时间再刷新；按 q 或 Esc 退出
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == 27:  # 27 = Esc
        break

cv2.destroyAllWindows()