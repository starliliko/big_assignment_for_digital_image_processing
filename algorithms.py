# -*- coding: utf-8 -*-
"""
图像去雾去雨算法模块
包含多种图像增强和物理模型方法
"""

import cv2
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter


class DehazeAlgorithms:
    """去雾算法集合"""
    
    @staticmethod
    def histogram_equalization(image):
        """
        直方图均衡化
        基于图像增强的方法，提升对比度
        """
        if len(image.shape) == 3:
            # 转换到YCrCb空间，只对亮度通道进行均衡化
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            result = cv2.equalizeHist(image)
        return result
    
    @staticmethod
    def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        对比度受限的自适应直方图均衡化 (CLAHE)
        基于图像增强的方法，避免过度增强噪声
        """
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        if len(image.shape) == 3:
            # 转换到LAB空间，对L通道进行CLAHE
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            result = clahe_obj.apply(image)
        return result
    
    @staticmethod
    def retinex_ssr(image, sigma=300):
        """
        单尺度Retinex算法 (SSR)
        基于图像增强的方法，模拟人眼视觉特性
        """
        image_float = image.astype(np.float64) + 1.0
        
        if len(image.shape) == 3:
            result = np.zeros_like(image_float)
            for i in range(3):
                # 高斯模糊估计光照
                blur = cv2.GaussianBlur(image_float[:, :, i], (0, 0), sigma)
                # Retinex: log(反射) = log(图像) - log(光照)
                result[:, :, i] = np.log10(image_float[:, :, i]) - np.log10(blur + 1.0)
        else:
            blur = cv2.GaussianBlur(image_float, (0, 0), sigma)
            result = np.log10(image_float) - np.log10(blur + 1.0)
        
        # 归一化到0-255
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)
    
    @staticmethod
    def retinex_msr(image, sigma_list=[15, 80, 250]):
        """
        多尺度Retinex算法 (MSR)
        基于图像增强的方法，结合多个尺度的效果
        """
        image_float = image.astype(np.float64) + 1.0
        
        if len(image.shape) == 3:
            result = np.zeros_like(image_float)
            for i in range(3):
                retinex = np.zeros_like(image_float[:, :, i])
                for sigma in sigma_list:
                    blur = cv2.GaussianBlur(image_float[:, :, i], (0, 0), sigma)
                    retinex += np.log10(image_float[:, :, i]) - np.log10(blur + 1.0)
                result[:, :, i] = retinex / len(sigma_list)
        else:
            result = np.zeros_like(image_float)
            for sigma in sigma_list:
                blur = cv2.GaussianBlur(image_float, (0, 0), sigma)
                result += np.log10(image_float) - np.log10(blur + 1.0)
            result /= len(sigma_list)
        
        # 归一化到0-255
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)
    
    @staticmethod
    def dark_channel_prior(image, omega=0.85, t0=0.1, patch_size=15):
        """
        暗通道先验去雾算法 (DCP) - 优化版
        基于物理模型的方法，何恺明经典算法
        
        大气散射模型: I(x) = J(x)*t(x) + A*(1-t(x))
        其中: I是有雾图像，J是无雾图像，t是透射率，A是大气光
        
        优化: 降低omega避免过度去雾，添加色彩增强
        """
        image_float = image.astype(np.float64) / 255.0
        
        # 1. 计算暗通道
        dark_channel = DehazeAlgorithms._get_dark_channel(image_float, patch_size)
        
        # 2. 估计大气光A
        atm_light = DehazeAlgorithms._estimate_atmospheric_light(image_float, dark_channel)
        
        # 3. 估计透射率t
        transmission = DehazeAlgorithms._estimate_transmission(image_float, atm_light, omega, patch_size)
        
        # 4. 导向滤波优化透射率
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        transmission = DehazeAlgorithms._guided_filter(gray, transmission, 60, 1e-3)
        
        # 5. 恢复无雾图像
        transmission = np.maximum(transmission, t0)
        result = np.zeros_like(image_float)
        for i in range(3):
            result[:, :, i] = (image_float[:, :, i] - atm_light[i]) / transmission + atm_light[i]
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        # 6. 色彩增强和对比度调整
        result = DehazeAlgorithms._enhance_result(result)
        
        return result
    
    @staticmethod
    def _get_dark_channel(image, patch_size):
        """计算暗通道"""
        min_channel = np.min(image, axis=2)
        dark_channel = minimum_filter(min_channel, size=patch_size)
        return dark_channel
    
    @staticmethod
    def _estimate_atmospheric_light(image, dark_channel, top_percent=0.001):
        """估计大气光"""
        h, w = dark_channel.shape
        num_pixels = int(max(h * w * top_percent, 1))
        
        # 选取暗通道中最亮的像素
        dark_flat = dark_channel.ravel()
        indices = np.argsort(dark_flat)[-num_pixels:]
        
        # 在这些像素中选择原图最亮的点作为大气光
        atm_light = np.zeros(3)
        for i in range(3):
            channel_flat = image[:, :, i].ravel()
            atm_light[i] = np.max(channel_flat[indices])
        
        return atm_light
    
    @staticmethod
    def _estimate_transmission(image, atm_light, omega, patch_size):
        """估计透射率"""
        normalized = np.zeros_like(image)
        for i in range(3):
            normalized[:, :, i] = image[:, :, i] / (atm_light[i] + 1e-6)
        
        dark_channel = DehazeAlgorithms._get_dark_channel(normalized, patch_size)
        transmission = 1 - omega * dark_channel
        return transmission
    
    @staticmethod
    def _guided_filter(guide, src, radius, eps):
        """导向滤波"""
        mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
        mean_src = cv2.boxFilter(src, -1, (radius, radius))
        mean_guide_src = cv2.boxFilter(guide * src, -1, (radius, radius))
        cov_guide_src = mean_guide_src - mean_guide * mean_src
        
        mean_guide_guide = cv2.boxFilter(guide * guide, -1, (radius, radius))
        var_guide = mean_guide_guide - mean_guide * mean_guide
        
        a = cov_guide_src / (var_guide + eps)
        b = mean_src - a * mean_guide
        
        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))
        
        return mean_a * guide + mean_b
    
    @staticmethod
    def _enhance_result(image):
        """
        结果增强：提升亮度和色彩饱和度
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 增强饱和度
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
        
        # 轻微提升亮度
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        
        # 转回BGR
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    @staticmethod
    def dark_channel_prior_adaptive(image):
        """
        自适应暗通道先验去雾算法
        根据图像雾浓度自动调整参数
        """
        image_float = image.astype(np.float64) / 255.0
        
        # 计算暗通道估计雾浓度
        dark_channel = DehazeAlgorithms._get_dark_channel(image_float, 15)
        fog_density = np.mean(dark_channel)
        
        # 根据雾浓度自适应调整omega
        if fog_density > 0.5:  # 重度雾
            omega = 0.90
            t0 = 0.15
        elif fog_density > 0.3:  # 中度雾
            omega = 0.85
            t0 = 0.1
        else:  # 轻度雾
            omega = 0.75
            t0 = 0.05
        
        # 估计大气光
        atm_light = DehazeAlgorithms._estimate_atmospheric_light(image_float, dark_channel)
        
        # 估计透射率
        transmission = DehazeAlgorithms._estimate_transmission(image_float, atm_light, omega, 15)
        
        # 导向滤波
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        transmission = DehazeAlgorithms._guided_filter(gray, transmission, 60, 1e-3)
        
        # 恢复图像
        transmission = np.maximum(transmission, t0)
        result = np.zeros_like(image_float)
        for i in range(3):
            result[:, :, i] = (image_float[:, :, i] - atm_light[i]) / transmission + atm_light[i]
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        # 增强结果
        result = DehazeAlgorithms._enhance_result(result)
        
        return result
    
    @staticmethod
    def gamma_correction(image, gamma=0.7):
        """
        伽马校正
        基于图像增强的方法，调整图像亮度
        """
        # 构建查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    @staticmethod
    def homomorphic_filter(image, gamma_l=0.5, gamma_h=2.0, c=1, d0=30):
        """
        同态滤波
        基于图像增强的方法，同时增强对比度和压缩动态范围
        """
        if len(image.shape) == 3:
            # 转换到YCrCb空间
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y_channel = ycrcb[:, :, 0].astype(np.float64)
        else:
            y_channel = image.astype(np.float64)
        
        # 取对数
        y_log = np.log1p(y_channel)
        
        # 傅里叶变换
        rows, cols = y_channel.shape
        dft = np.fft.fft2(y_log)
        dft_shift = np.fft.fftshift(dft)
        
        # 构建同态滤波器
        u = np.arange(rows)
        v = np.arange(cols)
        u, v = np.meshgrid(u - rows // 2, v - cols // 2, indexing='ij')
        d = np.sqrt(u ** 2 + v ** 2)
        h = (gamma_h - gamma_l) * (1 - np.exp(-c * (d ** 2) / (d0 ** 2))) + gamma_l
        
        # 滤波
        filtered = dft_shift * h
        
        # 逆变换
        idft_shift = np.fft.ifftshift(filtered)
        idft = np.fft.ifft2(idft_shift)
        result = np.expm1(np.real(idft))
        
        # 归一化
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if len(image.shape) == 3:
            ycrcb[:, :, 0] = result
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        
        return result


class DerainAlgorithms:
    """去雨算法集合"""
    
    @staticmethod
    def median_filter(image, ksize=5):
        """
        中值滤波去雨
        基于图像增强的方法，去除雨滴噪声
        """
        return cv2.medianBlur(image, ksize)
    
    @staticmethod
    def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
        """
        双边滤波去雨
        基于图像增强的方法，保边去噪
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    @staticmethod
    def guided_filter_derain(image, radius=16, eps=0.01):
        """
        导向滤波去雨
        基于图像增强的方法，平滑雨痕同时保留边缘
        """
        image_float = image.astype(np.float64) / 255.0
        
        if len(image.shape) == 3:
            result = np.zeros_like(image_float)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
            for i in range(3):
                result[:, :, i] = DehazeAlgorithms._guided_filter(gray, image_float[:, :, i], radius, eps)
        else:
            result = DehazeAlgorithms._guided_filter(image_float, image_float, radius, eps)
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result
    
    @staticmethod
    def morphological_derain(image, kernel_size=3, iterations=1):
        """
        形态学去雨
        基于图像增强的方法，利用开运算去除细小雨滴
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                # 开运算：先腐蚀后膨胀
                result[:, :, i] = cv2.morphologyEx(image[:, :, i], cv2.MORPH_OPEN, kernel, iterations=iterations)
        else:
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        return result
    
    @staticmethod
    def low_rank_derain(image, rank=10):
        """
        低秩分解去雨（简化版）
        基于图像增强的方法，分离背景和雨层
        """
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                channel = image[:, :, i].astype(np.float64)
                # SVD分解
                U, S, Vt = np.linalg.svd(channel, full_matrices=False)
                # 保留前rank个奇异值（背景）
                S[rank:] = 0
                reconstructed = np.dot(U, np.dot(np.diag(S), Vt))
                result[:, :, i] = np.clip(reconstructed, 0, 255).astype(np.uint8)
        else:
            U, S, Vt = np.linalg.svd(image.astype(np.float64), full_matrices=False)
            S[rank:] = 0
            result = np.clip(np.dot(U, np.dot(np.diag(S), Vt)), 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def dsc_derain(image, ksize=5, sigma=1.5):
        """
        基于稀疏编码的去雨（简化版）
        使用高斯差分和形态学操作
        """
        # 高斯模糊
        blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
        
        # 差分检测雨滴
        diff = cv2.absdiff(image, blurred)
        
        # 阈值处理
        if len(diff.shape) == 3:
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        else:
            diff_gray = diff
        
        _, mask = cv2.threshold(diff_gray, 20, 255, cv2.THRESH_BINARY)
        
        # 膨胀雨滴区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 使用inpaint修复雨滴区域
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        return result
