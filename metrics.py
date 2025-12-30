# -*- coding: utf-8 -*-
"""
图像质量评价指标模块
包含PSNR、SSIM、MAE、MSE等常用指标
"""

import cv2
import numpy as np
from scipy import ndimage


class ImageMetrics:
    """图像质量评价指标"""
    
    @staticmethod
    def mse(image1, image2):
        """
        均方误差 (Mean Squared Error)
        值越小表示图像越相似
        """
        image1 = image1.astype(np.float64)
        image2 = image2.astype(np.float64)
        return np.mean((image1 - image2) ** 2)
    
    @staticmethod
    def mae(image1, image2):
        """
        平均绝对误差 (Mean Absolute Error)
        值越小表示图像越相似
        """
        image1 = image1.astype(np.float64)
        image2 = image2.astype(np.float64)
        return np.mean(np.abs(image1 - image2))
    
    @staticmethod
    def psnr(image1, image2, max_val=255):
        """
        峰值信噪比 (Peak Signal-to-Noise Ratio)
        值越大表示图像质量越好，通常>30dB认为质量较好
        """
        mse_val = ImageMetrics.mse(image1, image2)
        if mse_val == 0:
            return float('inf')
        return 10 * np.log10((max_val ** 2) / mse_val)
    
    @staticmethod
    def ssim(image1, image2, window_size=11, k1=0.01, k2=0.03, L=255):
        """
        结构相似性指数 (Structural Similarity Index)
        值范围[-1, 1]，越接近1表示越相似
        """
        image1 = image1.astype(np.float64)
        image2 = image2.astype(np.float64)
        
        # 如果是彩色图像，转换为灰度图
        if len(image1.shape) == 3:
            image1 = cv2.cvtColor(image1.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        if len(image2.shape) == 3:
            image2 = cv2.cvtColor(image2.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2
        
        # 高斯窗口
        sigma = 1.5
        gaussian = cv2.getGaussianKernel(window_size, sigma)
        window = np.outer(gaussian, gaussian.transpose())
        
        # 计算均值
        mu1 = cv2.filter2D(image1, -1, window)
        mu2 = cv2.filter2D(image2, -1, window)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = cv2.filter2D(image1 ** 2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(image2 ** 2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(image1 * image2, -1, window) - mu1_mu2
        
        # SSIM公式
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)
    
    @staticmethod
    def rmse(image1, image2):
        """
        均方根误差 (Root Mean Squared Error)
        值越小表示图像越相似
        """
        return np.sqrt(ImageMetrics.mse(image1, image2))
    
    @staticmethod
    def entropy(image):
        """
        图像熵 (Entropy)
        衡量图像信息量，值越大表示信息越丰富
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算直方图
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # 去除零值
        hist = hist[hist > 0]
        
        # 计算熵
        return -np.sum(hist * np.log2(hist))
    
    @staticmethod
    def contrast(image):
        """
        对比度 (Contrast)
        衡量图像明暗对比程度
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return np.std(image.astype(np.float64))
    
    @staticmethod
    def average_gradient(image):
        """
        平均梯度 (Average Gradient)
        衡量图像清晰度，值越大表示越清晰
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = image.astype(np.float64)
        
        # 计算x和y方向的梯度
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        gradient = np.sqrt(gx ** 2 + gy ** 2)
        
        return np.mean(gradient)
    
    @staticmethod
    def sharpness(image):
        """
        锐度 (Sharpness)
        使用拉普拉斯算子衡量图像锐度
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return np.var(laplacian)
    
    @staticmethod
    def colorfulness(image):
        """
        色彩丰富度 (Colorfulness)
        衡量图像颜色的丰富程度
        """
        if len(image.shape) != 3:
            return 0.0
        
        # 分离通道
        B, G, R = cv2.split(image.astype(np.float64))
        
        # 计算rg和yb
        rg = R - G
        yb = 0.5 * (R + G) - B
        
        # 计算均值和标准差
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rg = np.mean(rg)
        mean_yb = np.mean(yb)
        
        # 计算色彩丰富度
        std_root = np.sqrt(std_rg ** 2 + std_yb ** 2)
        mean_root = np.sqrt(mean_rg ** 2 + mean_yb ** 2)
        
        return std_root + 0.3 * mean_root
    
    @staticmethod
    def niqe_approx(image):
        """
        自然图像质量评价（简化近似版）
        无参考图像质量评价，值越小越好
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = image.astype(np.float64)
        
        # 计算局部均值和方差
        mu = cv2.GaussianBlur(image, (7, 7), 7/6)
        sigma = np.sqrt(cv2.GaussianBlur(image**2, (7, 7), 7/6) - mu**2 + 1e-10)
        
        # 归一化
        normalized = (image - mu) / (sigma + 1e-10)
        
        # 计算统计特征
        mean_val = np.mean(normalized)
        var_val = np.var(normalized)
        
        # 简化的NIQE近似（基于偏离自然图像统计特性）
        return abs(mean_val) + abs(var_val - 1)
    
    @staticmethod
    def calculate_all_metrics(original, processed, include_no_reference=True):
        """
        计算所有评价指标
        
        Args:
            original: 原始图像（参考图像/清晰图像）
            processed: 处理后的图像
            include_no_reference: 是否包含无参考指标
        
        Returns:
            dict: 包含所有指标的字典
        """
        metrics = {}
        
        # 有参考指标（需要原始清晰图像）
        if original is not None:
            metrics['PSNR'] = ImageMetrics.psnr(original, processed)
            metrics['SSIM'] = ImageMetrics.ssim(original, processed)
            metrics['MSE'] = ImageMetrics.mse(original, processed)
            metrics['MAE'] = ImageMetrics.mae(original, processed)
            metrics['RMSE'] = ImageMetrics.rmse(original, processed)
        
        # 无参考指标
        if include_no_reference:
            metrics['熵'] = ImageMetrics.entropy(processed)
            metrics['对比度'] = ImageMetrics.contrast(processed)
            metrics['平均梯度'] = ImageMetrics.average_gradient(processed)
            metrics['锐度'] = ImageMetrics.sharpness(processed)
            if len(processed.shape) == 3:
                metrics['色彩丰富度'] = ImageMetrics.colorfulness(processed)
        
        return metrics
    
    @staticmethod
    def format_metrics(metrics):
        """
        格式化输出评价指标
        """
        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if value == float('inf'):
                    lines.append(f"{key}: ∞")
                else:
                    lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
