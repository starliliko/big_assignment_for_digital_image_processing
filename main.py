# -*- coding: utf-8 -*-
"""
图像去雾去雨系统 - 主程序
数字图像处理大作业
"""

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QGroupBox, QComboBox, QTextEdit, QScrollArea,
                             QMessageBox, QProgressBar, QTableWidget, QTableWidgetItem,
                             QHeaderView, QSplitter, QTabWidget, QSpinBox, QDoubleSpinBox,
                             QFormLayout, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from algorithms import DehazeAlgorithms, DerainAlgorithms
from metrics import ImageMetrics


class ImageLabel(QLabel):
    """可缩放的图像显示标签"""
    def __init__(self, title=""):
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                background-color: #f5f5f5;
                border-radius: 5px;
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
        
        self.init_ui()
        self.init_algorithms()
    
    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("图像去雾去雨系统 - 数字图像处理大作业")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #aaa;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #4a90d9;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2a5f8f;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
        """)
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧功能区
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 右侧显示区
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 4)
    
    def create_left_panel(self):
        """创建左侧功能面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        title_label = QLabel("图像去雾去雨系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("微软雅黑", 14, QFont.Bold))
        title_label.setStyleSheet("color: #333; padding: 10px;")
        layout.addWidget(title_label)
        
        info_label = QLabel("组名：zzZ\n成员：李彦博 焦浩洋 范凯纬")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(info_label)
        
        # 文件/系统组
        file_group = QGroupBox("文件/系统")
        file_layout = QVBoxLayout(file_group)
        
        btn_open = QPushButton("打开图像")
        btn_open.clicked.connect(self.open_image)
        file_layout.addWidget(btn_open)
        
        btn_open_gt = QPushButton("打开参考图(GT)")
        btn_open_gt.clicked.connect(self.open_gt_image)
        file_layout.addWidget(btn_open_gt)
        
        btn_save = QPushButton("保存结果")
        btn_save.clicked.connect(self.save_image)
        file_layout.addWidget(btn_save)
        
        btn_batch = QPushButton("批量处理")
        btn_batch.clicked.connect(self.batch_process)
        file_layout.addWidget(btn_batch)
        
        layout.addWidget(file_group)
        
        # 去雾算法组
        dehaze_group = QGroupBox("去雾算法")
        dehaze_layout = QVBoxLayout(dehaze_group)
        
        self.dehaze_combo = QComboBox()
        dehaze_layout.addWidget(self.dehaze_combo)
        
        btn_dehaze = QPushButton("执行去雾")
        btn_dehaze.clicked.connect(self.apply_dehaze)
        dehaze_layout.addWidget(btn_dehaze)
        
        layout.addWidget(dehaze_group)
        
        # 去雨算法组
        derain_group = QGroupBox("去雨算法")
        derain_layout = QVBoxLayout(derain_group)
        
        self.derain_combo = QComboBox()
        derain_layout.addWidget(self.derain_combo)
        
        btn_derain = QPushButton("执行去雨")
        btn_derain.clicked.connect(self.apply_derain)
        derain_layout.addWidget(btn_derain)
        
        layout.addWidget(derain_group)
        
        # 评价指标组
        metrics_group = QGroupBox("评价指标")
        metrics_layout = QVBoxLayout(metrics_group)
        
        btn_calculate = QPushButton("计算指标")
        btn_calculate.clicked.connect(self.calculate_metrics)
        metrics_layout.addWidget(btn_calculate)
        
        btn_compare = QPushButton("算法对比")
        btn_compare.clicked.connect(self.compare_algorithms)
        metrics_layout.addWidget(btn_compare)
        
        layout.addWidget(metrics_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """创建右侧显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 图像显示区
        image_splitter = QSplitter(Qt.Horizontal)
        
        # 原图
        original_group = QGroupBox("原图")
        original_layout = QVBoxLayout(original_group)
        self.original_label = ImageLabel("请打开图像")
        original_layout.addWidget(self.original_label)
        image_splitter.addWidget(original_group)
        
        # 结果图
        result_group = QGroupBox("结果图")
        result_layout = QVBoxLayout(result_group)
        self.result_label = ImageLabel("处理后显示")
        result_layout.addWidget(self.result_label)
        image_splitter.addWidget(result_group)
        
        layout.addWidget(image_splitter, 3)
        
        # 程序说明区
        info_group = QGroupBox("程序说明 / 评价指标输出")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("Consolas", 10))
        self.info_text.setPlainText(
            "欢迎使用图像去雾去雨系统！\n\n"
            "使用说明：\n"
            "1. 点击「打开图像」加载有雾/有雨图像\n"
            "2. 可选：点击「打开参考图(GT)」加载清晰参考图\n"
            "3. 选择去雾或去雨算法，点击执行\n"
            "4. 点击「计算指标」查看评价结果\n"
            "5. 点击「算法对比」进行多算法对比分析\n\n"
            "实现的算法：\n"
            "【去雾】直方图均衡化、CLAHE、自适应暗通道、伽马校正\n"
            "【去雨】中值滤波、双边滤波、导向滤波、形态学、低秩分解、稀疏编码\n\n"
            "评价指标：PSNR、SSIM、MSE、MAE、RMSE、熵、对比度、平均梯度、锐度、色彩丰富度"
        )
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group, 1)
        
        return panel
    
    def init_algorithms(self):
        """初始化算法列表"""
        # 去雾算法
        self.dehaze_algorithms = {
            "直方图均衡化 (HE)": DehazeAlgorithms.histogram_equalization,
            "CLAHE": DehazeAlgorithms.clahe,
            "自适应暗通道": DehazeAlgorithms.dark_channel_prior_adaptive,
            "伽马校正": DehazeAlgorithms.gamma_correction,
        }
        
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
    
    # 设置中文字体
    font = QFont("微软雅黑", 9)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
