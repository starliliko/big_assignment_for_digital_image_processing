# -*- coding: utf-8 -*-
"""
图像去雾去雨系统 - 主程序
数字图像处理大作业
"""

import sys
import os
import cv2
import numpy as np
from PyQt5 import uic  # 新增：使用 .ui 文件
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QGroupBox, QComboBox, QTextEdit, QScrollArea,
                             QMessageBox, QProgressBar, QTableWidget, QTableWidgetItem,
                             QHeaderView, QSplitter, QTabWidget, QSpinBox, QDoubleSpinBox,
                             QFormLayout, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont

from algorithms import DehazeAlgorithms, DerainAlgorithms
from metrics import ImageMetrics
# from Ui_main import Ui_MainWindow  # 不再使用自动生成的 Python UI 文件，直接加载 main.ui

class ImageLabel(QLabel):
    """可缩放的图像显示标签"""
    def __init__(self, parent=None, title=""):
        super().__init__(parent)
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        # 与整体界面统一的梦幻磨砂风格
        self.setStyleSheet("""
            QLabel {
                border-radius: 18px;
                border: 2px solid rgba(255, 255, 255, 0.65);
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 228, 246, 0.9),
                    stop:1 rgba(210, 235, 255, 0.9)
                );
                color: #333333;
            }
        """)
        self.setText(title if title else "无图像")
        self.original_pixmap = None
        
    def set_image(self, image):
        """设置显示图像（OpenCV格式）"""
        if image is None:
            self.setText(self.title if self.title else "无图像")
            self.original_pixmap = None
            return
            
        # 转换BGR到RGB
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        h, w = rgb_image.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.update_display()
    
    def update_display(self):
        """更新显示，自适应大小"""
        if self.original_pixmap:
            scaled = self.original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)
    
    def resizeEvent(self, event):
        """窗口大小改变时更新显示"""
        super().resizeEvent(event)
        self.update_display()


class BatchProcessThread(QThread):
    """批量处理线程"""
    progress = pyqtSignal(int)
    result = pyqtSignal(list)
    
    def __init__(self, image_paths, gt_paths, algorithm_func, algorithm_name):
        super().__init__()
        self.image_paths = image_paths
        self.gt_paths = gt_paths  # Ground truth paths
        self.algorithm_func = algorithm_func
        self.algorithm_name = algorithm_name
    
    def run(self):
        results = []
        total = len(self.image_paths)
        
        for i, img_path in enumerate(self.image_paths):
            try:
                # 读取图像
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # 处理图像
                processed = self.algorithm_func(image)
                
                # 如果有ground truth，计算有参考指标
                gt_image = None
                if i < len(self.gt_paths) and self.gt_paths[i]:
                    gt_image = cv2.imread(self.gt_paths[i])
                    if gt_image is not None and gt_image.shape != processed.shape:
                        gt_image = cv2.resize(gt_image, (processed.shape[1], processed.shape[0]))
                
                # 计算指标
                metrics = ImageMetrics.calculate_all_metrics(gt_image, processed)
                metrics['文件名'] = os.path.basename(img_path)
                metrics['算法'] = self.algorithm_name
                results.append(metrics)
                
            except Exception as e:
                print(f"处理 {img_path} 时出错: {e}")
            
            self.progress.emit(int((i + 1) / total * 100))
        
        self.result.emit(results)


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.processed_image = None
        self.gt_image = None  # Ground truth image
        self.batch_results = []

        # 窗口拖动相关状态
        self._is_dragging = False
        self._drag_pos = QPoint()
        
        # 使用 .ui 初始化界面
        self.init_ui()
        # 填充算法列表
        self.init_algorithms()
    
    def init_ui(self):
        """使用 QtDesigner 生成的 .ui 文件初始化界面"""
        ui_path = os.path.join(os.path.dirname(__file__), "main.ui")
        uic.loadUi(ui_path, self)

        # 设置左右区域伸缩比例为 1:3（左：右）
        main_layout = self.findChild(QHBoxLayout, "horizontalLayout")
        if main_layout is not None:
            main_layout.setStretch(0, 1)
            main_layout.setStretch(1, 3)

        # 左侧垂直布局：让按钮区域纵向更均匀铺满整列
        left_layout = self.findChild(QVBoxLayout, "verticalLayout_left")
        if left_layout is not None:
            # 0: 顶部自定义窗口按钮行
            # 1: 标题
            # 2: 组名/成员信息
            # 3-6: 各功能分组（尽量占据更多高度）
            # 7: 进度条
            # 8: 底部弹性空白
            left_layout.setStretch(0, 0)
            left_layout.setStretch(1, 0)
            left_layout.setStretch(2, 0)
            left_layout.setStretch(3, 1)
            left_layout.setStretch(4, 1)
            left_layout.setStretch(5, 1)
            left_layout.setStretch(6, 1)
            left_layout.setStretch(7, 0)
            left_layout.setStretch(8, 0)

        # 使用无边框窗口，自定义标题栏与控制按钮
        self.setWindowFlag(Qt.FramelessWindowHint, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # 用代码把 ui 里的占位 QLabel 替换成真正的 ImageLabel
        def _replace_with_imagelabel(obj_name: str) -> ImageLabel:
            placeholder = self.findChild(QLabel, obj_name)
            if placeholder is None:
                return None
            parent = placeholder.parent()
            layout = parent.layout()
            index = layout.indexOf(placeholder)
            layout.removeWidget(placeholder)
            placeholder.deleteLater()
            new_label = ImageLabel(parent=parent)
            new_label.setObjectName(obj_name)
            layout.insertWidget(index, new_label)
            return new_label

        self.original_label = _replace_with_imagelabel("lblOriginal")
        self.result_label = _replace_with_imagelabel("lblResult")

        # 绑定其它控件到原先使用的属性名
        self.info_text = self.findChild(QTextEdit, "textInfo")
        self.dehaze_combo = self.findChild(QComboBox, "comboDehaze")
        self.derain_combo = self.findChild(QComboBox, "comboDerain")
        self.progress_bar = self.findChild(QProgressBar, "progressBar")

        # 自定义窗口控制按钮
        self.btn_close = self.findChild(QPushButton, "btnClose")
        self.btn_minimize = self.findChild(QPushButton, "btnMinimize")
        self.btn_maximize = self.findChild(QPushButton, "btnMaximize")

        if self.btn_close is not None:
            self.btn_close.clicked.connect(self.close)
        if self.btn_minimize is not None:
            self.btn_minimize.clicked.connect(self.showMinimized)
        if self.btn_maximize is not None:
            self.btn_maximize.clicked.connect(self.toggle_max_restore)

        # 信号连接到已有槽函数
        self.findChild(QPushButton, "btn_openImage").clicked.connect(self.open_image)
        self.findChild(QPushButton, "btn_openGT").clicked.connect(self.open_gt_image)
        self.findChild(QPushButton, "btn_saveResult").clicked.connect(self.save_image)
        self.findChild(QPushButton, "btn_batchProcess").clicked.connect(self.batch_process)
        self.findChild(QPushButton, "btnApplyDehaze").clicked.connect(self.apply_dehaze)
        self.findChild(QPushButton, "btnApplyDerain").clicked.connect(self.apply_derain)
        self.findChild(QPushButton, "btnCalculateMetrics").clicked.connect(self.calculate_metrics)
        self.findChild(QPushButton, "btnCompareAlgorithms").clicked.connect(self.compare_algorithms)

        if self.progress_bar is not None:
            self.progress_bar.setVisible(False)

        # 放大整体字体
        font = QFont("微软雅黑", 11)
        self.setFont(font)

    def toggle_max_restore(self):
        """在最大化与还原之间切换"""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def mousePressEvent(self, event):
        """支持拖动无边框窗口"""
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging = False
        super().mouseReleaseEvent(event)

    def init_algorithms(self):
        """初始化算法列表"""
        # 去雾算法
        self.dehaze_algorithms = {
            "直方图均衡化 (HE)": DehazeAlgorithms.histogram_equalization,
            "CLAHE": DehazeAlgorithms.clahe,
            "自适应暗通道": DehazeAlgorithms.dark_channel_prior_adaptive,
            "伽马校正": DehazeAlgorithms.gamma_correction,
        }
        
        self.dehaze_combo.clear()
        for name in self.dehaze_algorithms.keys():
            self.dehaze_combo.addItem(name)
        
        # 去雨算法
        self.derain_algorithms = {
            "中值滤波": DerainAlgorithms.median_filter,
            "双边滤波": DerainAlgorithms.bilateral_filter,
            "导向滤波": DerainAlgorithms.guided_filter_derain,
            "形态学去雨": DerainAlgorithms.morphological_derain,
            "低秩分解": DerainAlgorithms.low_rank_derain,
            "稀疏编码去雨": DerainAlgorithms.dsc_derain,
        }
        
        self.derain_combo.clear()
        for name in self.derain_algorithms.keys():
            self.derain_combo.addItem(name)
    
    def open_image(self):
        """打开图像文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "", 
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        
        if file_path:
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.original_label.set_image(self.current_image)
                self.processed_image = None
                self.result_label.set_image(None)
                self.info_text.append(f"\n已加载图像: {os.path.basename(file_path)}")
                self.info_text.append(f"图像尺寸: {self.current_image.shape[1]}x{self.current_image.shape[0]}")
            else:
                QMessageBox.warning(self, "错误", "无法读取图像文件！")
    
    def open_gt_image(self):
        """打开参考图像（Ground Truth）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开参考图像", "", 
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;所有文件 (*)"
        )
        
        if file_path:
            self.gt_image = cv2.imread(file_path)
            if self.gt_image is not None:
                self.info_text.append(f"\n已加载参考图(GT): {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "错误", "无法读取参考图像！")
    
    def save_image(self):
        """保存处理结果"""
        if self.processed_image is None:
            QMessageBox.warning(self, "提示", "没有可保存的处理结果！")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "result.png",
            "PNG文件 (*.png);;JPEG文件 (*.jpg);;所有文件 (*)"
        )
        
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            self.info_text.append(f"\n结果已保存: {file_path}")
    
    def apply_dehaze(self):
        """应用去雾算法"""
        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先打开图像！")
            return
        
        algorithm_name = self.dehaze_combo.currentText()
        algorithm_func = self.dehaze_algorithms[algorithm_name]
        
        self.info_text.append(f"\n正在应用去雾算法: {algorithm_name}...")
        
        try:
            self.processed_image = algorithm_func(self.current_image)
            self.result_label.set_image(self.processed_image)
            self.info_text.append("处理完成！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败: {str(e)}")
            self.info_text.append(f"错误: {str(e)}")
    
    def apply_derain(self):
        """应用去雨算法"""
        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先打开图像！")
            return
        
        algorithm_name = self.derain_combo.currentText()
        algorithm_func = self.derain_algorithms[algorithm_name]
        
        self.info_text.append(f"\n正在应用去雨算法: {algorithm_name}...")
        
        try:
            self.processed_image = algorithm_func(self.current_image)
            self.result_label.set_image(self.processed_image)
            self.info_text.append("处理完成！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理失败: {str(e)}")
            self.info_text.append(f"错误: {str(e)}")
    
    def calculate_metrics(self):
        """计算评价指标"""
        if self.processed_image is None:
            QMessageBox.warning(self, "提示", "请先处理图像！")
            return
        
        # 调整GT图像尺寸（如果存在）
        gt_for_metrics = None
        if self.gt_image is not None:
            if self.gt_image.shape != self.processed_image.shape:
                gt_for_metrics = cv2.resize(self.gt_image, 
                    (self.processed_image.shape[1], self.processed_image.shape[0]))
            else:
                gt_for_metrics = self.gt_image
        
        # 计算指标
        metrics = ImageMetrics.calculate_all_metrics(gt_for_metrics, self.processed_image)
        
        # 显示结果
        self.info_text.append("\n" + "="*50)
        self.info_text.append("评价指标结果:")
        self.info_text.append("="*50)
        
        if gt_for_metrics is not None:
            self.info_text.append("【有参考指标】")
        else:
            self.info_text.append("【无参考指标】（未加载GT图像）")
        
        self.info_text.append(ImageMetrics.format_metrics(metrics))
        self.info_text.append("="*50)
    
    def compare_algorithms(self):
        """对比多种算法"""
        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先打开图像！")
            return
        
        self.info_text.append("\n" + "="*60)
        self.info_text.append("算法对比分析")
        self.info_text.append("="*60)
        
        # 调整GT图像尺寸
        gt_for_metrics = None
        if self.gt_image is not None:
            sample_result = self.dehaze_algorithms["直方图均衡化 (HE)"](self.current_image)
            if self.gt_image.shape != sample_result.shape:
                gt_for_metrics = cv2.resize(self.gt_image, 
                    (sample_result.shape[1], sample_result.shape[0]))
            else:
                gt_for_metrics = self.gt_image
        
        results = []
        
        # 测试去雾算法
        self.info_text.append("\n【去雾算法对比】")
        for name, func in self.dehaze_algorithms.items():
            try:
                processed = func(self.current_image)
                metrics = ImageMetrics.calculate_all_metrics(gt_for_metrics, processed)
                metrics['算法'] = name
                results.append(metrics)
                
                if gt_for_metrics is not None:
                    self.info_text.append(
                        f"{name}: PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}, "
                        f"熵={metrics['熵']:.2f}"
                    )
                else:
                    self.info_text.append(
                        f"{name}: 熵={metrics['熵']:.2f}, 对比度={metrics['对比度']:.2f}, "
                        f"平均梯度={metrics['平均梯度']:.2f}"
                    )
            except Exception as e:
                self.info_text.append(f"{name}: 处理失败 - {str(e)}")
        
        # 测试去雨算法
        self.info_text.append("\n【去雨算法对比】")
        for name, func in self.derain_algorithms.items():
            try:
                processed = func(self.current_image)
                metrics = ImageMetrics.calculate_all_metrics(gt_for_metrics, processed)
                metrics['算法'] = name
                results.append(metrics)
                
                if gt_for_metrics is not None:
                    self.info_text.append(
                        f"{name}: PSNR={metrics['PSNR']:.2f}, SSIM={metrics['SSIM']:.4f}, "
                        f"熵={metrics['熵']:.2f}"
                    )
                else:
                    self.info_text.append(
                        f"{name}: 熵={metrics['熵']:.2f}, 对比度={metrics['对比度']:.2f}, "
                        f"平均梯度={metrics['平均梯度']:.2f}"
                    )
            except Exception as e:
                self.info_text.append(f"{name}: 处理失败 - {str(e)}")
        
        self.info_text.append("="*60)
        self.batch_results = results
    
    def batch_process(self):
        """批量处理图像"""
        # 选择图像文件夹
        folder = QFileDialog.getExistingDirectory(self, "选择有雾/有雨图像文件夹")
        if not folder:
            return
        
        # 获取图像文件列表
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_paths = []
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in image_extensions:
                image_paths.append(os.path.join(folder, f))
        
        if not image_paths:
            QMessageBox.warning(self, "提示", "文件夹中没有找到图像文件！")
            return
        
        # 可选：选择GT文件夹
        reply = QMessageBox.question(self, "Ground Truth", 
            "是否有对应的清晰参考图像(GT)文件夹？",
            QMessageBox.Yes | QMessageBox.No)
        
        gt_paths = []
        if reply == QMessageBox.Yes:
            gt_folder = QFileDialog.getExistingDirectory(self, "选择GT图像文件夹")
            if gt_folder:
                for img_path in image_paths:
                    base_name = os.path.basename(img_path)
                    gt_path = os.path.join(gt_folder, base_name)
                    if os.path.exists(gt_path):
                        gt_paths.append(gt_path)
                    else:
                        gt_paths.append(None)
        
        # 选择算法
        algorithm_name = self.dehaze_combo.currentText()
        algorithm_func = self.dehaze_algorithms[algorithm_name]
        
        self.info_text.append(f"\n开始批量处理 {len(image_paths)} 张图像...")
        self.info_text.append(f"使用算法: {algorithm_name}")
        
        # 启动批量处理线程
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.batch_thread = BatchProcessThread(
            image_paths, gt_paths, algorithm_func, algorithm_name
        )
        self.batch_thread.progress.connect(self.progress_bar.setValue)
        self.batch_thread.result.connect(self.on_batch_complete)
        self.batch_thread.start()
    
    def on_batch_complete(self, results):
        """批量处理完成"""
        self.progress_bar.setVisible(False)
        self.batch_results = results
        
        if not results:
            self.info_text.append("批量处理完成，但没有有效结果。")
            return
        
        self.info_text.append("\n" + "="*70)
        self.info_text.append("批量处理结果统计")
        self.info_text.append("="*70)
        
        # 计算平均指标
        metric_keys = ['PSNR', 'SSIM', 'MSE', 'MAE', '熵', '对比度', '平均梯度']
        averages = {}
        
        for key in metric_keys:
            values = [r[key] for r in results if key in r and r[key] != float('inf')]
            if values:
                averages[key] = np.mean(values)
        
        self.info_text.append(f"\n处理图像数量: {len(results)}")
        self.info_text.append("\n平均指标:")
        for key, value in averages.items():
            self.info_text.append(f"  {key}: {value:.4f}")
        
        self.info_text.append("\n详细结果:")
        for r in results:
            line = f"  {r.get('文件名', 'N/A')}: "
            if 'PSNR' in r:
                line += f"PSNR={r['PSNR']:.2f}, SSIM={r['SSIM']:.4f}"
            else:
                line += f"熵={r.get('熵', 0):.2f}"
            self.info_text.append(line)
        
        self.info_text.append("="*70)


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置中文字体（整体字号稍大一些）
    font = QFont("微软雅黑", 11)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
