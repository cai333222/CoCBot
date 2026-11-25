import cv2
import numpy as np
import time
import uiautomator2 as u2


def multi_scale_template_match(big_img, template_img, threshold=0.7):
    """多尺度模板匹配，在一定缩放范围内找模板"""
    h_t, w_t = template_img.shape[:2]
    h_b, w_b = big_img.shape[:2]

    # 尝试缩放范围 0.3 ~0.8倍
    for scale in np.linspace(0.3, 0.8, 15):
        resized = cv2.resize(template_img, None, fx=scale, fy=scale)
        r_h, r_w = resized.shape[:2]
        if r_h > h_b or r_w > w_b:
            continue

        res = cv2.matchTemplate(big_img, resized, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        pts = list(zip(*loc[::-1]))
        if len(pts) > 0:
            print(f"匹配成功 scale={scale:.2f}")
            x1, y1 = pts[0]
            x2 = x1 + r_w
            y2 = y1 + r_h
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            return True, (int(x), int(y))
    return False, (0, 0)


def wait_template_match(d, template, threshold=0.7, timeout=15, roi=None):
    """
    循环截图做模板匹配，直到匹配成功或者超时
    :param d: u2设备对象
    :param template: 模板图片(numpy数组)
    :param threshold: 匹配阈值
    :param timeout: 超时秒
    :param roi: (x1,y1,x2,y2)局部搜索区域
    :return: (ok, pos)
    """
    start = time.time()
    while time.time() - start < timeout:
        pil_img = d.screenshot()
        screen = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if roi is not None:
            x1, y1, x2, y2 = roi
            screen = screen[y1:y2, x1:x2]

        ok, pos = multi_scale_template_match(screen, template, threshold=threshold)
        if ok:
            print(f"✅模板匹配成功，坐标:{pos}")
            return ok, pos
        time.sleep(1)
    print(f"❌模板匹配超时 {timeout}s")
    return False, None


def run_one_battle(d, template_pika, template_jiqi, template_jieshu):
    """执行一次完整战斗流程"""
    POS_SEARCH = (212, 1900)
    POS_ATTACK = (2422, 1357)

    # 点击搜索+攻击
    d.click(POS_SEARCH[0], POS_SEARCH[1])
    d.click(POS_ATTACK[0], POS_ATTACK[1])
    time.sleep(1)

    # 等待皮卡
    ok, POS_PIKA_SELECT = wait_template_match(d, template_pika, threshold=0.7, timeout=60)
    if not ok:
        raise RuntimeError("皮卡模板匹配超时失败")

    # 等待机器
    ok, POS_JIQI_SELECT = wait_template_match(d, template_jiqi, threshold=0.7, timeout=15)
    if not ok:
        raise RuntimeError("机器模板匹配超时失败")
    print(f"机器坐标 {POS_JIQI_SELECT}")

    # 点击皮卡 + 下兵
    d.click(POS_PIKA_SELECT[0], POS_PIKA_SELECT[1])
    click_count = 6
    for i in range(click_count):
        d.click(2857, 947)
    d.click(POS_JIQI_SELECT[0], POS_JIQI_SELECT[1])
    d.click(2857, 947)

    # 等待战斗结束模板
    ok, pos_end = wait_template_match(d, template_jieshu, threshold=0.7, timeout=120)
    d.click(pos_end[0], pos_end[1])
    print("战斗结束模板结果", ok, pos_end)
    if not ok:
        raise RuntimeError("等待战斗结束超时")

    # 等待页面退回搜索界面，根据你的手机速度调大
    print("✅本局战斗完成，准备下一局")
    time.sleep(1)


if __name__ == "__main__":
    # ========== 设备连接 ==========
    DEVICE_IP = "192.168.0.103"
    d = u2.connect(DEVICE_IP)
    print("✅设备已连接")

    # ========== 加载模板（全局一次性加载） ==========
    template_nvwu = cv2.imread("test_img/nvwu.png")
    template_jiqi = cv2.imread("test_img/jiqi.png")
    template_jieshu = cv2.imread("test_img/jieshu.png")
    template_source = cv2.imread("test_img/source.png")

    # 简单校验模板是否读取成功
    for name, tpl in [("nvwu", template_nvwu), ("jiqi", template_jiqi), ("jieshu", template_jieshu), ("source", template_source)]:
        if tpl is None:
            raise FileNotFoundError(f"模板读取失败 test_img/{name}.png")

    round_num = 1
    while True:
        print(f"\n========== 开始第 {round_num} 局 ==========")
        try:
            run_one_battle(d, template_nvwu, template_jiqi, template_jieshu)

            # ====== 每5局执行一次滑动+模板匹配逻辑 ======
            if round_num % 5 == 0:
                print(f"-----已跑{round_num}局，执行5次一轮的特殊操作-----")
                d.swipe(1700, 500, 700, 1000, 0.1)
                time.sleep(0.5)
                pil_screen = d.screenshot()
                screen = cv2.cvtColor(np.array(pil_screen), cv2.COLOR_RGB2BGR)

                ok, pos = multi_scale_template_match(screen, template_source, threshold=0.7)
                print(ok, pos)

                if ok:
                    x = int(pos[0])
                    y = int(pos[1])
                    cv2.circle(screen, (x, y), 12, (0, 0, 255), -1)
                    cv2.putText(screen, f"CLICK:{x},{y}", (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.imwrite("test_img/mark_result.png", screen)
                    print("✅已生成标记图片 test_img/mark_result.png")
                    d.click(x, y)
                else:
                    print("❌未匹配到模板，不执行点击")

                d.click(2413, 1769)
                d.click(2791, 260)
                time.sleep(1)

            round_num += 1

        except Exception as e:
            print(f"⚠️本局异常：{e}，休眠5秒后继续尝试下一局")
            time.sleep(5)
