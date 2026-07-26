# 图像去雾与去雨系统

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-41CD52?logo=qt&logoColor=white)](https://www.riverbankcomputing.com/software/pyqt/)
[![Tests](https://github.com/starliliko/image-dehaze-derain-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/starliliko/image-dehaze-derain-toolkit/actions/workflows/tests.yml)

数字图像处理课程大作业：一个基于传统图像处理方法的桌面端去雾、去雨与质量评价工具。项目将多种算法、参考/无参考评价指标、批量处理和可视化界面整合在一个 PyQt5 应用中，便于观察不同方法的效果与适用场景。

> 本项目用于课程学习和算法对比。传统增强方法可以改善部分图像的视觉效果，但不能替代面向真实复杂天气训练的深度学习恢复模型。

## 项目亮点

- 4 种界面可选去雾方法，包含增强类与物理模型类算法
- 6 种界面可选去雨方法，覆盖空间滤波、形态学和低秩分解
- 支持单张处理、算法横向对比和文件夹批处理
- 支持中文及包含空格的图像路径
- 同时提供 5 项有参考指标与 5 项无参考指标
- UI 与算法、指标模块分离，便于独立实验和扩展

## 功能概览

| 类别 | 方法 |
| --- | --- |
| 去雾 | 直方图均衡化、CLAHE、自适应暗通道先验、伽马校正 |
| 去雨 | 中值滤波、双边滤波、导向滤波、形态学处理、低秩分解、稀疏编码近似 |
| 有参考指标 | PSNR、SSIM、MSE、MAE、RMSE |
| 无参考指标 | 熵、对比度、平均梯度、锐度、色彩丰富度 |
| 工作方式 | 单张处理、算法对比、批量处理、结果保存 |

`algorithms.py` 中还保留了 Retinex、同态滤波和基础暗通道先验等实现，可用于继续扩展界面选项或开展独立实验。

## 快速开始

### 1. 获取项目

```bash
git clone https://github.com/starliliko/image-dehaze-derain-toolkit.git
cd image-dehaze-derain-toolkit
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
source .venv/bin/activate
```

### 3. 安装并运行

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

建议使用 Python 3.9–3.12。首次启动需要安装带桌面 GUI 支持的 PyQt5；在无图形界面的服务器环境中只能运行算法与测试。

## 使用流程

1. 点击“打开图像”，载入有雾或有雨图像。
2. 如有清晰参考图，点击“打开参考图（GT）”。
3. 选择去雾或去雨算法并执行。
4. 点击“计算指标”查看量化结果，或使用“算法对比”横向比较。
5. 保存结果；处理数据集时可使用“批量处理”。

程序通过 `numpy.fromfile` 与 `cv2.imdecode` 读取文件，并通过 `cv2.imencode` 保存结果，因此可以处理中文路径和带空格路径。

## 项目结构

```text
.
├── main.py          # PyQt5 主程序、文件操作与批处理线程
├── main.ui          # Qt Designer 界面定义
├── Ui_main.py       # Qt Designer 生成的 Python 界面代码
├── algorithms.py    # 去雾与去雨算法
├── metrics.py       # 有参考与无参考评价指标
├── tests/           # 核心算法与指标测试
└── requirements.txt
```

运行时数据流：

```text
输入图像 ──→ 去雾/去雨算法 ──→ 处理结果 ──→ 保存
    │                              │
    └──── 可选 Ground Truth ────────┴──→ 质量指标
```

## 算法说明

### 自适应暗通道先验

暗通道先验从大气散射模型出发：

```text
I(x) = J(x)t(x) + A(1 - t(x))
```

其中 `I` 为观测图像，`J` 为待恢复图像，`t` 为透射率，`A` 为大气光。项目根据图像雾浓度调整参数，并在恢复后进行颜色增强。

### CLAHE

在局部区域执行带对比度限制的直方图均衡化。它速度快、结果稳定，并能减少普通直方图均衡化放大噪声的问题。

### 去雨方法

本项目主要使用传统图像分解与平滑思路抑制雨纹，包括保边滤波、方向结构处理、形态学操作以及低秩近似。不同方法会在雨纹抑制、边缘保留和计算速度之间产生不同权衡。

## 评价指标

- PSNR、SSIM、MSE、MAE、RMSE 需要尺寸对应的清晰参考图。
- 熵、对比度、平均梯度、锐度和色彩丰富度无需参考图。
- 单一指标不能完整代表视觉质量，建议结合原图、处理结果和多项指标判断。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试使用程序生成的小型合成图像，不依赖外部数据集。GitHub Actions 会在每次推送和 Pull Request 时运行相同测试。

## 可扩展方向

- 为 Retinex、同态滤波等已有实现增加界面入口
- 引入 RESIDE、Rain100 等公开数据集，建立统一基准
- 记录每种算法的运行耗时与批量统计结果
- 增加参数调节面板和处理前后滑动对比
- 接入轻量级深度学习模型，与传统方法进行对照

## 参考资料

- He, K., Sun, J., & Tang, X. *Single Image Haze Removal Using Dark Channel Prior*.
- [RESIDE 去雾数据集](https://sites.google.com/view/reside-dehaze-datasets)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Qt for Python / Qt Documentation](https://doc.qt.io/)

## 许可

项目当前未声明开源许可证，仅用于学习与课程展示。若计划允许他人复制、修改或再发布，请在确认团队成员意见后补充合适的许可证。
