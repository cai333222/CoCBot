import cv2
import numpy as np
import time
import uiautomator2 as u2

def multi_scale_template_match(big_img, template_img, threshold=0.7):
    """多尺度模板匹配，在一定缩放范围内找模板"""
    h_t, w_t = template_img.shape[:2]
    h_b, w_b = big_img.shape[:2]

    # 尝试缩放范围 0.5 ~0.8倍，根据你的截图微调
    for scale in np.linspace(0.5, 0.8, 5):
        resized = cv2.resize(template_img, None, fx=scale, fy=scale)
        r_h, r_w = resized.shape[:2]
        if r_h > h_b or r_w > w_b:
            continue

        res = cv2.matchTemplate(big_img, resized, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        pts = list(zip(*loc[::-1]))
        if len(pts) > 0:
            x1, y1 = pts[0]
            x2 = x1 + r_w
            y2 = y1 + r_h
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            return True, (x, y)
    return False, (0, 0)

if __name__ == "__main__":
    # d.screenshot("test_img/screenshot.png")
    template_queen = cv2.imread("test_img/queen.png")
    template_yongwang = cv2.imread("test_img/yongwang.png")
    template_cangying = cv2.imread("test_img/CangYing.png")
    template_longwang = cv2.imread("test_img/LongWang.png")
    template_leilong = cv2.imread("test_img/leilong.png")
    template_zhuzi = cv2.imread("test_img/zhuzi.png")
    screen = cv2.imread("test_img/screenshot.png")
    # ok, box_queen = multi_scale_template_match(screen, template_queen, threshold=0.7)
    # ok, box_yongwang = multi_scale_template_match(screen, template_yongwang, threshold=0.7)
    # ok, box_cangying = multi_scale_template_match(screen, template_cangying, threshold=0.7)
    # ok, box_longwang = multi_scale_template_match(screen, template_longwang, threshold=0.7)
    # print(box_queen, box_yongwang, box_cangying, box_longwang)
    # d.click(box_queen[0]+(box_queen[2]-box_queen[0])/2, box_queen[1]+(box_queen[3]-box_queen[1])/2)
    # time.sleep(0.1)
    # d.click(box_yongwang[0]+(box_yongwang[2]-box_yongwang[0])/2, box_yongwang[1]+(box_yongwang[3]-box_yongwang[1])/2)
    # time.sleep(0.1)
    # d.click(box_cangying[0]+(box_cangying[2]-box_cangying[0])/2, box_cangying[1]+(box_cangying[3]-box_cangying[1])/2)
    # time.sleep(0.1)
    # d.click(box_longwang[0]+(box_longwang[2]-box_longwang[0])/2, box_longwang[1]+(box_longwang[3]-box_longwang[1])/2) 
    d = u2.connect("192.168.0.103")
    ok, box_leilong = multi_scale_template_match(screen, template_leilong, threshold=0.7)
    ok, box_zhuzi = multi_scale_template_match(screen, template_zhuzi, threshold=0.7)
    print(box_leilong,box_zhuzi)
    d.click(box_zhuzi[0]+(box_zhuzi[2]-box_zhuzi[0])/2, box_zhuzi[1]+(box_zhuzi[3]-box_zhuzi[1])/2)
    time.sleep(0.1)
    d.click(box_leilong[0]+(box_leilong[2]-box_leilong[0])/2, box_leilong[1]+(box_leilong[3]-box_leilong[1])/2)