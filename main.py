# ===================== 【必须放在所有import最顶部】屏蔽Paddle日志 =====================
import os
import warnings
# os.environ["GLOG_minloglevel"] = "2"
# os.environ["PADDLE_HIDE_INITIAL_MESSAGE"] = "1"
# os.environ["FLAGS_call_stack_level"] = "2"
# warnings.filterwarnings("ignore")
# ==================================================================================

import uiautomator2 as u2
import time
import sys
from datetime import datetime as dt
import cv2
import numpy as np
from orb_match import multi_scale_template_match
from paddlex import create_model

SCRIPT_START_TIME = time.perf_counter()
PHASE_TIMES = []


def record_phase(name):
    elapsed = time.perf_counter() - SCRIPT_START_TIME
    PHASE_TIMES.append((name, elapsed))
    print(f"⏱️ [启动耗时] {name}: {elapsed:.2f}s")

# ==============================================
#               【统一常量配置区】
#      所有坐标、颜色、IP、开关、阈值在此修改
# ==============================================
# -------- 设备与应用 --------
DEVICE_IP = "192.168.0.103"
PACKAGE_NAME = "com.tencent.tmgp.supercell.clashofclans"
# =====================【全局只加载一次模型】=====================
DET_MODEL_PATH = r"D:\coc_bat\models\det\best_accuracy\inference"
REC_MODEL_PATH = r"D:\coc_bat\models\rec\best_accuracy\inference"

record_phase("脚本启动，开始初始化")
model_init_start = time.perf_counter()
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
record_phase(f"模型加载完成（共耗时 {(time.perf_counter() - model_init_start):.2f}s）")
# =================================================================
# -------- 界面点击坐标 --------
POS_SEARCH = (212, 1900)
POS_ATTACK = (553, 1505)
POS_JINGONG = (2672, 1714)
POS_YUANBING = (1575, 1167)
POS_QINGQIU = (1143, 1723)
POS_FASONG = (1987, 1426)

# 战斗结束/退出相关坐标
POS_EXIT_BATTLE = (229, 1618)
POS_CONFIRM_LEAVE = (1963, 1304)
POS_CLOSE_SETTLE = (1589, 1814)

# -------- 资源识别颜色 RGB --------
COLOR_GOLD_FULL = (244, 220, 113)    # 金币满色
COLOR_ELIXIR_FULL = (226, 143, 225)  # 圣水满色
COLOR_OIL_FULL = (39, 14, 51)          # 黑油满色

# 资源颜色检测坐标
POS_GOLD_CHECK = (2565, 86)
POS_ELIXIR_CHECK = (2612, 228)
POS_OIL_CHECK = (2708, 394)
SAVE_DEBUG_IMG = False   # False=完全不写图片到磁盘；True调试时保存截图
# -------- 等待识别点位与目标颜色 --------
# 战斗加载完成检测点
POS_BATTLE_LOAD_CHECK = (2862, 1476)
COLOR_BATTLE_LOAD = (252, 188, 54)
# 战斗结束按钮检测点
POS_BATTLE_END_CHECK = (1467, 1787)
COLOR_BATTLE_END = (216, 245, 119)


# -------- 识别参数（颜色容差、超时） --------
COLOR_TOLERANCE = 10
WAIT_COLOR_TIMEOUT = 10

# -------- 文件相关 --------
timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
IMG_FOLDER = f"dataset/{timestamp}_img"

# -------- 战斗下兵点位【统一收拢】 --------
# 雷龙起始结束点位、投放数量
template_queen = cv2.imread("test_img/queen.png")
template_yongwang = cv2.imread("test_img/yongwang.png")
template_cangying = cv2.imread("test_img/CangYing.png")
template_longwang = cv2.imread("test_img/LongWang.png")
template_leilong = cv2.imread("test_img/leilong.png")
template_zhuzi = cv2.imread("test_img/zhuzi.png")
template_feiting = cv2.imread("test_img/feiting.png")
template_manwang = cv2.imread("test_img/manwang.png")
template_runtu = cv2.imread("test_img/runtu.png")
DRAGON_START = (1619, 1704)
DRAGON_END = (2712, 880)
DRAGON_CLICK_NUM = 11
POS_DRAGON_SELECT = None

# 英雄坐标
POS_QUEEN_SELECT = None #女王
POS_QUEEN_DROP = (2675, 821)

POS_WARDEN_SELECT = None #永王
POS_WARDEN_DROP = (2015, 1322)

POS_ROYAL_CHAMP_SELECT = None #蛮王
POS_ROYAL_CHAMP_DROP = (2015, 1322)

POS_GIANT_GUARD_SELECT = None #闰土
POS_GIANT_GUARD_DROP = (2015, 1322)

POS_BALLOON_SELECT = None#飞艇
POS_BALLOON_DROP = (2015, 1322)


# 攻城训练营选中坐标
# POS_SIEGE_CAMP_SELECT = (1354, 1833)
# 柱子投放批次坐标列表
SIEGE_BATCHES = [
    [(1498, 1346), (1921, 1054), (2276, 817)],
    [(1409, 1154), (1757, 938), (2090, 733)],
    [(1313, 1007), (1630, 812), (1908, 639)],
    [(1170, 906), (1467, 716), (1746, 537)],
    [(905, 779), (1208, 534), (1560, 295)]
]

# -------- 基础延时参数（后续可改为随机区间） --------
DELAY_SHORT = 0.3
DELAY_WAIT_TROOP = 16
DELAY_LOOP_INTERVAL = 0.5
DELAY_CAMP_INTERVAL = 5

# ==============================================
#                   日志输出类
# ==============================================
class TeeLogger:
    def __init__(self, log_path):
        self.console = sys.stdout
        self.log_file = open(log_path, "a", encoding="utf-8")

    def write(self, message):
        self.console.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

# ==============================================
#                   初始化
# ==============================================
print(f"\n{'='*50}")
print(f"🚀 脚本启动时间: {dt.now()}")
print(f"🔧 目标设备 IP: {DEVICE_IP}")
print(f"{'='*50}\n")

try:
    connect_start = time.perf_counter()
    d = u2.connect(DEVICE_IP)
    record_phase(f"设备连接完成（耗时 {(time.perf_counter() - connect_start):.2f}s）")
    print(f"✅ 设备连接成功: {d.info['productName']}")
    print(f"📱 设备分辨率: {d.window_size()}")
    print(f"🌐 设备 IP: {d.wlan_ip}")
except Exception as e:
    print(f"❌ [致命错误] 连接失败: {e}")
    exit()

# 创建截图文件夹
if not os.path.exists(IMG_FOLDER):
    folder_start = time.perf_counter()
    os.makedirs(IMG_FOLDER)
    record_phase(f"创建截图/日志目录（耗时 {(time.perf_counter() - folder_start):.2f}s）")

# 日志文件路径：放在截图文件夹内
LOG_PATH = os.path.join(IMG_FOLDER, "run.log")
logger = TeeLogger(LOG_PATH)
sys.stdout = logger
print(f"📝 日志文件保存路径：{LOG_PATH}")
record_phase("日志系统初始化完成")

# 读取命令行参数控制蛋糕模式
arg_start = time.perf_counter()
if len(sys.argv) >= 2:
    try:
        USE_CAKE = int(sys.argv[1])
        if USE_CAKE not in (0, 1):
            raise ValueError
    except Exception:
        print("❌ 参数错误！只能传入 0 或 1")
        print("👉 使用示例：python coc.py 1")
        input("按回车退出...")
        logger.close()
        exit(1)
else:
    print("❌ 缺少启动参数！")
    print("👉 格式：python coc脚本.py 1  开启蛋糕")
    print("👉 格式：python coc脚本.py 0  关闭蛋糕")
    input("按回车退出...")
    logger.close()
    exit(1)
record_phase(f"参数解析完成（耗时 {(time.perf_counter() - arg_start):.2f}s）")
print(f"🎂 蛋糕模式: {'启用' if USE_CAKE == 1 else '关闭'}")

# ==============================================
#                   工具函数
# ==============================================
def safe_click(pos, info=""):
    if not isinstance(pos, tuple) or len(pos) != 2:
        print(f"❌ [点击失败] 坐标格式无效: {pos} | 说明={info}")
        return False
    x, y = pos
    w, h = d.window_size()
    if not (0 <= x <= w and 0 <= y <= h):
        print(f"⚠️ [点击越界] 目标 {info} 坐标 ({x}, {y}) 超出屏幕范围 ({w}x{h})，但仍将尝试点击")
    try:
        print(f"👉 [点击] {info or '动作'} -> ({x}, {y})")
        d.click(x, y)
        return True
    except Exception as e:
        print(f"❌ [点击失败] {info or '动作'} -> ({x}, {y})：{e}")
        return False


def request_troops():
    if USE_CAKE == 0:
        print("📩 开始发起援兵请求...")
        steps = [
            (POS_YUANBING, "援兵按钮"),
            (POS_QINGQIU, "请求支援"),
            (POS_FASONG, "发送请求")
        ]
        for pos, name in steps:
            if safe_click(pos, name):
                time.sleep(DELAY_SHORT)
            else:
                print(f"⚠️ [警告] {name} 点击失败，继续下一步")
        print("📩 援兵请求流程结束，等待支援填充...")
        time.sleep(DELAY_WAIT_TROOP)
    else:
        print("📌 蛋糕模式开启，跳过援兵请求流程")


# --------【新增配置，放到统一常量配置区】--------

def take_screenshot(count):
    print(f"📸 [资源识别] 开始截图并识别第 {count} 次战斗后的资源情况...")
    try:
        # 1、截图，全程内存，不写磁盘
        pil_img = d.screenshot()
        # PIL RGB → OpenCV BGR
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h_origin, w_origin = img.shape[:2]
        print(f"📷 [资源识别] 截图尺寸: {w_origin}x{h_origin}")

        # 调试模式才保存文件
        if SAVE_DEBUG_IMG:
            filepath = os.path.join(IMG_FOLDER, f"fish_{count}.png")
            cv2.imwrite(filepath, img)
            print(f"✅ 调试截图已保存：{filepath}")

        # 裁剪坐标不变
        y1, y2 = 880, 1300
        x1, x2 = 1000, 1800

        # 边界保护
        y2 = min(y2, h_origin)
        x2 = min(x2, w_origin)
        crop_area = img[y1:y2, x1:x2]

        if crop_area.size == 0:
            print("❌ [资源识别] 裁剪区域为空！坐标超出图片范围")
            return {"gold": 0, "elixir": 0, "dark_elixir": 0}

        print(f"🔎 [资源识别] 开始检测文本区域，裁剪范围: x={x1}-{x2}, y={y1}-{y2}")
        det_result = list(det_model.predict(crop_area))[0]
        dt_polys = det_result["dt_polys"]
        print(f"📊 [资源识别] 检测到 {len(dt_polys)} 个候选文本框")

        candidate_list = []
        for poly in dt_polys:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            xmin = int(min(xs))
            ymin = int(min(ys))
            xmax = int(max(xs))
            ymax = int(max(ys))

            crop_text = crop_area[ymin:ymax, xmin:xmax]
            rec_out = list(rec_model.predict(crop_text))[0]
            text = rec_out["rec_text"].strip()
            rec_score = rec_out["rec_score"]

            candidate_list.append({
                "text": text,
                "num": text,
                "score": rec_score,
                "y_pos": ymin
            })

        # 从上到下排序
        candidate_list.sort(key=lambda x: x["y_pos"])
        valid_nums = [item["num"] for item in candidate_list]

        resource_data = {
            "gold": valid_nums[0] if len(valid_nums) >= 1 else 0,
            "elixir": valid_nums[1] if len(valid_nums) >= 2 else 0,
            "dark_elixir": valid_nums[2] if len(valid_nums) >= 3 else 0
        }

        print(f"\n📊 [资源识别结果] 🪙金币:{resource_data['gold']} 💧圣水:{resource_data['elixir']} 🛢️黑油:{resource_data['dark_elixir']}")
        if len(valid_nums) < 2:
            print(f"⚠️ [资源识别] 有效识别数量只有{len(valid_nums)}条，识别异常，候选值={valid_nums}")

        return resource_data

    except Exception as e:
        print(f"❌ [资源识别] 截图/识别发生异常：{e}")
        import traceback
        traceback.print_exc()
        return {"gold": 0, "elixir": 0, "dark_elixir": 0}


def get_pixel_color(x, y):
    try:
        screenshot = d.screenshot()
        color = screenshot.convert("RGB").getpixel((x, y))
        return color
    except Exception as e:
        print(f"❌ [取色] 获取颜色失败: {e}")
        return (0, 0, 0)

def get_multi_pixels(pos_list):
    """
    一次截图，批量取多个像素RGB颜色
    :param pos_list: list[(x,y)]
    :return: list[(r,g,b)]，顺序和输入坐标一一对应
    """
    try:
        pil_img = d.screenshot()
        rgb_img = pil_img.convert("RGB")
        result = []
        for (x, y) in pos_list:
            result.append(rgb_img.getpixel((x, y)))
        return result
    except Exception as e:
        print(f"❌ [批量取色]截图失败: {e}")
        return [(0,0,0)] * len(pos_list)


def wait_for_color(x, y, target_color, timeout=WAIT_COLOR_TIMEOUT, threshold=COLOR_TOLERANCE, desc="目标状态"):
    start_time = time.time()
    print(f"⏳ [等待] {desc} | 坐标=({x}, {y}) | 目标RGB={target_color} | 超时={timeout}s | 容差={threshold}")
    while time.time() - start_time < timeout:
        current_color = get_pixel_color(x, y)
        diff = sum((c1 - c2) ** 2 for c1, c2 in zip(current_color, target_color)) ** 0.5
        if diff <= threshold:
            print(f"✅ [完成] {desc} 匹配成功！当前RGB={current_color}，差异值={diff:.2f}")
            time.sleep(0.5)
            return True
        if int(time.time() - start_time) % 2 == 0 and int(time.time() - start_time) > 0:
            print(f"⏱️ [等待中] {desc} 已等待 {time.time() - start_time:.1f}s，当前RGB={current_color}，差异={diff:.2f}")
        time.sleep(0.2)
    print(f"⏰ [超时] {desc} 未在 {timeout}s 内达到目标状态，坐标=({x}, {y})")
    return False


def is_color_match(color1, color2, tolerance=COLOR_TOLERANCE):
    match_result = all(abs(c1 - c2) <= tolerance for c1, c2 in zip(color1, color2))
    status = "✅ 资源已满" if match_result else "❌ 资源未满"
    print(f"🔍 [颜色检测] 当前 {color1} | 目标 {color2} | 容差 {tolerance} | 结果: {status}")
    return match_result

# ==============================================
#                   战斗逻辑
# ==============================================
def attack_village(count):
    global POS_DRAGON_SELECT, POS_QUEEN_SELECT, POS_WARDEN_SELECT
    global POS_GIANT_GUARD_SELECT, POS_ROYAL_CHAMP_SELECT, POS_SIEGE_CAMP_SELECT, POS_BALLOON_SELECT
    print(f"\n🚀 [作战] 开始第 {count} 次进攻流程")
    print("➡️ [阶段1] 进入搜索/攻击页")
    safe_click(POS_SEARCH, "搜索")
    time.sleep(DELAY_SHORT)
    safe_click(POS_ATTACK, "攻击")
    print("🔍 [阶段1] 已进入搜索列表")
    time.sleep(0.2)
    if USE_CAKE == 1:
        print("🎂 [蛋糕模式] 执行蛋糕相关点击")
        d.click(2610, 1567)
        time.sleep(0.1)
        d.click(1976, 1450)
    safe_click(POS_JINGONG, "进攻")
    print("⚔️ [阶段2] 已发起正式进攻")

    # 等待战斗加载完成
    wait_for_color(POS_BATTLE_LOAD_CHECK[0], POS_BATTLE_LOAD_CHECK[1], COLOR_BATTLE_LOAD, desc="战斗加载完成")
    if(count == 1):
        print("🔍 [模板匹配] 第一次进攻，获取所有部队和训练营坐标")
        pil_screen = d.screenshot()
        screen = cv2.cvtColor(np.array(pil_screen), cv2.COLOR_RGB2BGR)
        ok, POS_DRAGON_SELECT = multi_scale_template_match(screen, template_leilong, threshold=0.7)
        ok, POS_QUEEN_SELECT = multi_scale_template_match(screen, template_queen, threshold=0.7)
        ok, POS_WARDEN_SELECT = multi_scale_template_match(screen, template_yongwang, threshold=0.7)
        ok, POS_GIANT_GUARD_SELECT = multi_scale_template_match(screen, template_runtu, threshold=0.7)
        ok, POS_ROYAL_CHAMP_SELECT = multi_scale_template_match(screen, template_cangying, threshold=0.7)
        ok, POS_SIEGE_CAMP_SELECT = multi_scale_template_match(screen, template_zhuzi, threshold=0.7)
        ok, POS_BALLOON_SELECT = multi_scale_template_match(screen, template_feiting, threshold=0.7)
        print("✅ [模板匹配] 结果：", {
            "雷龙": POS_DRAGON_SELECT,
            "女王": POS_QUEEN_SELECT,
            "永王": POS_WARDEN_SELECT,
            "闰土": POS_GIANT_GUARD_SELECT,
            "蛮王": POS_ROYAL_CHAMP_SELECT,
            "柱子": POS_SIEGE_CAMP_SELECT,
            "飞艇": POS_BALLOON_SELECT,
        })
    else:
        print(f"♻️ [模板匹配] 第 {count} 次已复用之前的部队坐标")

    print("➡️ [阶段3] 开始下兵，先滑动到布阵区域")
    d.swipe(1700, 1500, 1000, 1000, 0.1)

    # --- 雷龙投放 ---
    print("🦎 [投兵] 雷龙投放开始")
    d.click(POS_DRAGON_SELECT[0], POS_DRAGON_SELECT[1])
    start_x, start_y = DRAGON_START
    end_x, end_y = DRAGON_END
    click_count = DRAGON_CLICK_NUM
    step_x = (end_x - start_x) / (click_count - 1)
    step_y = (end_y - start_y) / (click_count - 1)
    for i in range(click_count):
        click_x = int(start_x + i * step_x)
        click_y = int(start_y + i * step_y)
        d.click(click_x, click_y)
    print(f"✅ [投兵] 雷龙投放完成，共点击 {click_count} 次")

    # --- 英雄投放 ---
    print("👑 [投兵] 处理英雄单位")
    d.click(POS_QUEEN_SELECT[0], POS_QUEEN_SELECT[1])
    d.click(POS_QUEEN_DROP[0], POS_QUEEN_DROP[1])
    d.click(POS_QUEEN_SELECT[0], POS_QUEEN_SELECT[1])
    d.click(POS_WARDEN_SELECT[0], POS_WARDEN_SELECT[1])
    d.click(POS_WARDEN_DROP[0], POS_WARDEN_DROP[1])
    d.click(POS_GIANT_GUARD_SELECT[0], POS_GIANT_GUARD_SELECT[1])
    d.click(POS_GIANT_GUARD_DROP[0], POS_GIANT_GUARD_DROP[1])
    d.click(POS_ROYAL_CHAMP_SELECT[0], POS_ROYAL_CHAMP_SELECT[1])
    d.click(POS_ROYAL_CHAMP_DROP[0], POS_ROYAL_CHAMP_DROP[1])
    d.click(POS_ROYAL_CHAMP_SELECT[0], POS_ROYAL_CHAMP_SELECT[1])
    d.click(POS_BALLOON_SELECT[0], POS_BALLOON_SELECT[1])
    d.click(POS_BALLOON_DROP[0], POS_BALLOON_DROP[1])
    print("✅ [投兵] 英雄投放已完成")

    # --- 攻城训练营（柱子）---
    print("🏹 [投兵] 开始投放攻城训练营单位")
    d.click(POS_SIEGE_CAMP_SELECT[0], POS_SIEGE_CAMP_SELECT[1])
    for i, batch in enumerate(SIEGE_BATCHES):
        print(f"  - 批次 {i+1}/{len(SIEGE_BATCHES)}：点击 {len(batch)} 个落点")
        for x, y in batch:
            d.click(x, y)
        if i == 0:
            d.click(POS_WARDEN_SELECT[0], POS_WARDEN_SELECT[1])
            d.click(POS_GIANT_GUARD_SELECT[0], POS_GIANT_GUARD_SELECT[1])
            d.click(POS_SIEGE_CAMP_SELECT[0], POS_SIEGE_CAMP_SELECT[1])
        if i < 4:
            time.sleep(DELAY_CAMP_INTERVAL)
    print("✅ [投兵] 攻城训练营投放结束")

    # --- 战斗结束判断 ---
    print("➡️ [阶段4] 等待战斗结束按钮出现")
    if wait_for_color(POS_BATTLE_END_CHECK[0], POS_BATTLE_END_CHECK[1], COLOR_BATTLE_END, desc="结束战斗按钮出现"):
        print("🏠 [回营] 战斗结束，已返回村庄")
        take_screenshot(count)
    else:
        print("⚠️ [回营] 战斗未正常结束，尝试强制退出战斗")
        safe_click(POS_EXIT_BATTLE, "结束战斗")
        time.sleep(0.2)
        safe_click(POS_CONFIRM_LEAVE, "确认离开")
        time.sleep(0.5)
        safe_click(POS_CLOSE_SETTLE, "关闭结算")
        print("🏠 [回营] 已强制回村并执行资源识别")
        take_screenshot(count)

# ==============================================
#                   主循环
# ==============================================
if __name__ == "__main__":
    try:
        start = time.time()
        count = 0
        while True:
            count += 1
            if count == 1:
                first_loop_elapsed = time.perf_counter() - SCRIPT_START_TIME
                print(f"\n⏱️ [启动总耗时] 程序从启动到第一次进入主循环共耗时: {first_loop_elapsed:.2f}s")
                print("📊 [启动阶段明细]")
                for phase_name, phase_elapsed in PHASE_TIMES:
                    print(f"  - {phase_name}: {phase_elapsed:.2f}s")
            print(f"\n{'='*60}")
            print(f"📅 当前时间: {dt.now().strftime('%H:%M:%S')}")
            print(f"🚀 开始第 {count} 次打鱼循环")
            print(f"{'='*60}")

            start_time = time.time()
            try:
                #request_troops()
                print("▶️ [主循环] 执行攻击/回营动作")
                attack_village(count)
            except Exception as e:
                print(f"\n❌ [致命错误] 战斗循环中出现异常: {e}")

            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️ [统计] 本次循环耗时: {duration:.2f} 秒")
            print(f"{'='*60}\n")
            time.sleep(DELAY_LOOP_INTERVAL)

            # 检测资源是否已满
            print("🔎 [主循环] 开始检查资源是否已满")
            pixel_positions = [
                POS_GOLD_CHECK,
                POS_ELIXIR_CHECK,
                POS_OIL_CHECK
            ]
            time.sleep(0.5)  # 等待截图稳定
            current_jinbi, current_shengshui, current_you = get_multi_pixels(pixel_positions)
            print(f"📍 [主循环] 资源检测像素位置: 金币={POS_GOLD_CHECK}, 圣水={POS_ELIXIR_CHECK}, 黑油={POS_OIL_CHECK}")

            if (is_color_match(current_jinbi, COLOR_GOLD_FULL) and is_color_match(current_shengshui, COLOR_ELIXIR_FULL) and is_color_match(current_you, COLOR_OIL_FULL)):
                total_seconds = time.time() - start
                minutes = int(total_seconds // 60)
                seconds = total_seconds % 60
                print(f"⏱️ [资源判定] 资源已满，准备升级城墙，总执行时间：{minutes}分{seconds:.1f}秒")
                break
            else:
                print("ℹ️ [资源判定] 当前资源未满，继续下一轮循环")

    except KeyboardInterrupt:
        print(f"\n\n🛑 检测到用户中断 (Ctrl+C)，脚本安全停止。")
    except Exception as e:
        print(f"\n\n💥 发生未捕获异常: {e}")
    finally:
        logger.close()
        print(f"\n📝 日志已保存至：{LOG_PATH}")