# CoC Auto Bot
## 项目概述
本项目是针对 Supercell 手游《部落冲突(Clash of Clans)》开发的自动化打鱼脚本原型。
运行主机仅支持 Windows，受控设备为安卓手机，依靠图像识别与远程设备控制实现全自动刷资源流程。

## 依赖环境
- Python 3
- uiautomator2：安卓设备远程连接、截图、模拟点击滑动
- OpenCV + Numpy：图像处理、画面裁剪、像素颜色检测
- PaddleX PP-OCRv5：文字检测与数字识别，读取金币、圣水、黑油数值
- PyYAML：读取统一配置文件 `config.yaml`
- 安卓设备开启无线 ADB 调试，电脑与手机处于同一局域网

安装依赖：
```bash
pip install -r requirements.txt
```

## 配置
所有坐标、颜色、设备 IP、阈值、模板路径、延时都集中在 `config.yaml` 中，运行前只需改这一处：

- `device.ip`：安卓设备无线 ADB IP（**先改这里**）
- `device.package_name`：部落冲突包名
- `models.*`：OCR 检测/识别模型路径与推理设备（gpu/cpu）
- `positions.*`：界面点击坐标
- `colors.*` / `resource_check.*`：资源满仓颜色与检测点
- `battle.*`：战斗加载/结束状态检测点
- `templates.*`：模板文件（key 为统一命名，value 为 `test_img/` 下的文件名）
- `dragon.*` / `hero_drop.*` / `hero_drop_order` / `siege_batches`：下兵点位与顺序
- `ability_trigger_units`：下兵后**立即**多点一次触发技能的单位（女王/蛮王/亡灵王子）。永王、闰土不在此列表，它们在攻城段（部队推进时）重新点选触发。
- `delays.*`：各类延时
- `crop.*`：资源数字裁剪区域

> 所有坐标基于设备分辨率 3200×2136。换手机/分辨率时需用 `get_coords.py` 重新采集并同步修改。

## 工作原理
1. 加载 `config.yaml`，初始化日志输出模块，加载 PP-OCR 文字检测、文字识别推理模型；
2. 通过 uiautomator2 连接局域网安卓设备（IP 来自配置），校验设备连通状态；
3. 进入循环自动化流程，重复执行完整打鱼流程；
4. 单次循环执行逻辑：
   1. 自动搜索可进攻村庄，进入对战界面；
   2. 等待战斗界面加载完成，按配置坐标自动投放雷龙、英雄、攻城器械；
   3. 检测战斗结束标识，自动退出战斗并关闭结算弹窗；
   4. 截取游戏画面，使用 PP-OCR 识别资源数字，保存截图与运行日志；
   5. 像素取色检测金币、圣水、黑油存储是否已满；
5. 三种资源全部存满后自动终止脚本运行。

## 核心功能
1. 远程设备操控：uiautomator2 下发点击、滑动指令，实时截取手机屏幕；
2. 界面颜色判定：通过 RGB 像素匹配判断加载完成、战斗结束、资源满仓状态；
3. OCR 数字识别：裁剪资源栏画面，识别金币、圣水、黑油存量数值；
4. 全自动战斗：内置雷龙、英雄、攻城器械全套投放点位，支持多批次投放；
5. 日志存档：每轮循环自动生成截图文件与完整运行日志，记录识别结果、运行耗时、报错信息；
6. 功能开关：启动参数控制蛋糕模式，0 关闭、1 开启。

## 启动命令
```bash
# 关闭蛋糕模式
python main.py 0

# 开启蛋糕模式
python main.py 1
```

## 目录结构
```
.
├── main.py            # 主脚本（雷龙流）
├── config.yaml        # 统一配置（坐标/颜色/IP/阈值/模板路径）
├── orb_match.py       # 多尺度模板匹配
├── get_coords.py      # 坐标采集工具（在截图窗口点击输出坐标）
├── night_main.py      # 夜世界打鱼脚本（皮卡流）
├── coc_ocr/           # OCR 推理与图片裁剪预处理
├── dataset/           # 运行截图、OCR 训练数据
├── models/            # OCR 检测/识别模型产物
└── test_img/          # 模板图片
```

## 英雄/单位命名对照
| 统一命名 | 中文 | 模板文件 |
|---|---|---|
| `hero_queen` | 女王 | `test_img/queen.png` |
| `hero_warden` | 永王 | `test_img/YongWang.png` |
| `hero_king` | 蛮王 | `test_img/manwang.png` |
| `hero_royal_champ` | 闰土 | `test_img/runtu.png` |
| `hero_minion_prince` | 亡灵王子（Minion Prince） | `test_img/CangYing.png` |
| `dragon` | 雷龙 | `test_img/leilong.png` |
| `siege_camp` | 柱子（攻城训练营） | `test_img/zhuzi.png` |
| `blimp` | 飞艇 | `test_img/feiting.png` |
