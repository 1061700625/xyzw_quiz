import logging
from typing import Callable, Any, List, Tuple
import difflib
import cv2
from matplotlib import pyplot as plt
import csv
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pygetwindow as gw
import mss
import time
import tkinter as tk
from tkinter import ttk, messagebox
from paddleocr import PaddleOCR
import pyautogui
from fuzzywuzzy import process
from paddleocr import paddleocr
paddleocr.logging.disable(logging.DEBUG)

import database

def require_window(method: Callable) -> Callable:
    """
    装饰器，用于检查是否已设置窗口。如果未设置窗口，则显示警告信息。

    参数:
    method (Callable): 需要装饰的函数。

    返回:
    Callable: 装饰后的函数。
    """
    def wrapper(self: 'WindowHandler', *args: Any, **kwargs: Any) -> Any:
        if not self.window:
            self.show_message("请先设置窗口。", "warning")
            return
        return method(self, *args, **kwargs)
    return wrapper

class WinHandler:
    def __init__(self):
        """
        初始化WindowHandler实例，设置初始窗口为None。
        """
        self.window = None

    def list_windows(self) -> List[str]:
        """
        获取所有窗口的标题列表。

        返回:
        List[str]: 窗口标题列表。
        """
        windows = gw.getAllTitles()
        return [title for title in windows if title]  # 忽略空标题

    def choose_window(self) -> None:
        """
        显示窗口选择对话框，供用户选择窗口。
        """
        windows = self.list_windows()
        if not windows:
            self.show_message("未找到任何窗口。", "error")
            return False
        root = tk.Tk()
        root.title("选择窗口")
        # 设置窗口大小和位置
        window_width = 400
        window_height = 250
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
        root.configure(bg="#F5F5F5")
        # 添加标签
        label = tk.Label(root, text="请选择一个窗口标题：", font=("Arial", 12), bg="#F5F5F5")
        label.pack(pady=20)
        # 添加下拉菜单
        selected_window = tk.StringVar()
        dropdown = ttk.Combobox(root, textvariable=selected_window, values=windows, state='readonly', font=("Arial", 10))
        dropdown.pack(pady=10, padx=20, fill=tk.X)
        dropdown.current(0)
        # 确认按钮回调函数
        def on_select() -> None:
            window_title = selected_window.get()
            for win in gw.getWindowsWithTitle(window_title):
                if win.title == window_title:
                    self.window = win
                    break
            if not self.window:
                self.show_message("未找到指定窗口，请确保窗口标题正确。", "error")
            root.destroy()
        # 添加确认按钮
        button = tk.Button(root, text="确认", command=on_select, font=("SimHei", 12), bg="#4CAF50", fg="white", relief="flat", height=2)
        button.pack(pady=20, padx=20, fill=tk.X)
        # 设置窗口风格
        style = ttk.Style()
        # 使用主题
        style.theme_use("clam")  
        # 调整下拉菜单内边距
        style.configure("TCombobox", padding=5)  
        root.mainloop()
        return True

    @require_window
    def capture_screenshot(self, filename: str = "screenshot.png") -> str:
        """
        捕获当前窗口的截图并保存为文件。

        参数:
        filename (str): 截图文件的保存路径。

        返回:
        str: 截图文件的保存路径。
        """
        self.window.activate()
        time.sleep(1)
        left, top, right, bottom = self.window.left, self.window.top, self.window.right, self.window.bottom
        with mss.mss() as sct:
            monitor = {"top": top, "left": left, "width": right-left, "height": bottom-top}
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            # 保存截图
            img.save(filename)
        return filename

    @require_window
    def move_and_resize_window(self, x: int, y: int, width: int, height: int) -> None:
        """
        移动并调整窗口大小。

        参数:
        x (int): 窗口的新X坐标。
        y (int): 窗口的新Y坐标。
        width (int): 窗口的新宽度。
        height (int): 窗口的新高度。
        """
        self.window.moveTo(x, y)
        self.window.resizeTo(width, height)

    @require_window
    def minimize_window(self) -> None:
        """
        最小化窗口。
        """
        self.window.minimize()

    @require_window
    def maximize_window(self) -> None:
        """
        最大化窗口。
        """
        self.window.maximize()

    @require_window
    def restore_window(self) -> None:
        """
        还原窗口。
        """
        self.window.restore()

    @require_window
    def close_window(self) -> None:
        """
        关闭窗口。
        """
        self.window.close()

    @require_window
    def focus_window(self) -> None:
        """
        激活并聚焦窗口。
        """
        self.window.activate()
        self.window.restore()

    def show_message(self, message: str, msg_type: str = "info") -> None:
        """
        显示消息框。

        参数:
        message (str): 要显示的消息内容。
        msg_type (str): 消息类型，可以是"info"、"warning"或"error"。
        """
        root = tk.Tk()
        root.withdraw()
        if msg_type == "info":
            messagebox.showinfo("信息", message)
        elif msg_type == "warning":
            messagebox.showwarning("警告", message)
        elif msg_type == "error":
            messagebox.showerror("错误", message)
        root.destroy()

class WinOperator:
    def __init__(self, window=None):
        """
        初始化WinOperator类。

        参数:
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则操作将在整个桌面上进行。
        """
        self.window = window

    def click(self, x: int, y: int, window=None) -> None:
        """
        在指定窗口中的特定位置执行点击操作。

        参数:
        x (int): 相对于窗口左上角的X坐标。
        y (int): 相对于窗口左上角的Y坐标。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        self._perform_action('click', x, y, window)

    def double_click(self, x: int, y: int, window=None) -> None:
        """
        在指定窗口中的特定位置执行双击操作。

        参数:
        x (int): 相对于窗口左上角的X坐标。
        y (int): 相对于窗口左上角的Y坐标。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        self._perform_action('doubleClick', x, y, window)

    def long_click(self, x: int, y: int, duration: float = 2.0, window=None) -> None:
        """
        在指定窗口中的特定位置执行长按操作。

        参数:
        x (int): 相对于窗口左上角的X坐标。
        y (int): 相对于窗口左上角的Y坐标。
        duration (float, optional): 长按的持续时间。默认是2秒。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        self._perform_long_click_action(x, y, duration, window)
        
    def move_to(self, x: int, y: int, window=None) -> None:
        """
        将鼠标移动到指定窗口中的特定位置。

        参数:
        x (int): 相对于窗口左上角的X坐标。
        y (int): 相对于窗口左上角的Y坐标。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        self._perform_action('moveTo', x, y, window)

    def right_click(self, x: int, y: int, window=None) -> None:
        """
        在指定窗口中的特定位置执行右击操作。

        参数:
        x (int): 相对于窗口左上角的X坐标。
        y (int): 相对于窗口左上角的Y坐标。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        self._perform_action('rightClick', x, y, window)

    def scroll(self, clicks: int, window=None) -> None:
        """
        在指定窗口内执行滚动操作。

        参数:
        clicks (int): 滚动的行数。如果为正数，则向上滚动；如果为负数，则向下滚动。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        self._perform_scroll_action(clicks, window)
        
    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5, window=None) -> None:
        """
        在指定窗口内执行滑动操作。

        参数:
        start_x (int): 起始点相对于窗口左上角的X坐标。
        start_y (int): 起始点相对于窗口左上角的Y坐标。
        end_x (int): 终点相对于窗口左上角的X坐标。
        end_y (int): 终点相对于窗口左上角的Y坐标。
        duration (float, optional): 滑动的持续时间。默认是0.5秒。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        self._perform_swipe_action(start_x, start_y, end_x, end_y, duration, window)

    def gesture(self, points: list, duration: float = 0.5, window=None) -> None:
        """
        在指定窗口内执行多个坐标点的手势操作。

        参数:
        points (list): 坐标点列表，每个点是一个元组 (x, y)，相对于窗口左上角的坐标。
        duration (float, optional): 每次移动的持续时间。默认是0.5秒。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        self._perform_gesture_action(points, duration, window)
        
    def _perform_action(self, action: str, x: int, y: int, window=None) -> None:
        """
        执行指定的鼠标操作。

        参数:
        action (str): 要执行的操作名称（如'click', 'doubleClick', 'moveTo', 'rightClick'）。
        x (int): 相对于窗口左上角的X坐标。
        y (int): 相对于窗口左上角的Y坐标。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        target_window = window if window else self.window
        if target_window:
            target_window.activate()
            window_left, window_top = target_window.left, target_window.top
            getattr(pyautogui, action)(window_left + x, window_top + y)
        else:
            getattr(pyautogui, action)(x, y)
    
    def _perform_scroll_action(self, clicks: int, window=None) -> None:
        """
        执行滚动操作。

        参数:
        clicks (int): 滚动的行数。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        target_window = window if window else self.window
        if target_window:
            target_window.activate()
            pyautogui.scroll(clicks)
        else:
            pyautogui.scroll(clicks)
    
    def _perform_swipe_action(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float, window=None) -> None:
        """
        执行滑动操作。

        参数:
        start_x (int): 起始点相对于窗口左上角的X坐标。
        start_y (int): 起始点相对于窗口左上角的Y坐标。
        end_x (int): 终点相对于窗口左上角的X坐标。
        end_y (int): 终点相对于窗口左上角的Y坐标。
        duration (float): 滑动的持续时间。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        target_window = window if window else self.window
        if target_window:
            target_window.activate()
            window_left, window_top = target_window.left, target_window.top
            pyautogui.moveTo(window_left + start_x, window_top + start_y)
            pyautogui.dragTo(window_left + end_x, window_top + end_y, duration=duration)
        else:
            pyautogui.moveTo(start_x, start_y)
            pyautogui.dragTo(end_x, end_y, duration=duration)

    def _perform_gesture_action(self, points: list, duration: float, window=None) -> None:
        """
        执行手势操作。

        参数:
        points (list): 坐标点列表，每个点是一个元组 (x, y)。
        duration (float): 每次移动的持续时间。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        target_window = window if window else self.window
        if target_window:
            target_window.activate()
            window_left, window_top = target_window.left, target_window.top
            for (x, y) in points:
                pyautogui.moveTo(window_left + x, window_top + y, duration=duration)
        else:
            for (x, y) in points:
                pyautogui.moveTo(x, y, duration=duration)
                
    def _perform_long_click_action(self, x: int, y: int, duration: float, window=None) -> None:
        """
        执行长按操作。

        参数:
        x (int): 相对于窗口左上角的X坐标。
        y (int): 相对于窗口左上角的Y坐标。
        duration (float): 长按的持续时间。
        window (object, optional): 通过pygetwindow获取的窗口对象。如果未提供，则使用实例化时的窗口。
        """
        target_window = window if window else self.window
        if target_window:
            target_window.activate()
            window_left, window_top = target_window.left, target_window.top
            pyautogui.mouseDown(window_left + x, window_top + y)
            pyautogui.sleep(duration)
            pyautogui.mouseUp(window_left + x, window_top + y)
        else:
            pyautogui.mouseDown(x, y)
            pyautogui.sleep(duration)
            pyautogui.mouseUp(x, y)

class Ocr:
    def __init__(self) -> None:
        self.ocr = PaddleOCR()
        self.data = None  # 存储OCR识别结果
    
    def multi_scale_template_match(self, main_image_path, template_image_path, method=cv2.TM_CCOEFF_NORMED, threshold=0.5, show=False):
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # 定义要使用的尺度列表
        # 加载图像
        main_image = cv2.imread(main_image_path)
        template_image = cv2.imread(template_image_path)
        
        # 初始化变量以存储最佳匹配信息
        best_match = None
        best_val = -np.inf if method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else np.inf
        
        # 在多个尺度下进行匹配
        for scale in scales:
            scaled_template = cv2.resize(template_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # 拆分颜色通道
            main_b, main_g, main_r = cv2.split(main_image)
            template_b, template_g, template_r = cv2.split(scaled_template)
            
            # 对每个通道进行模板匹配
            result_b = cv2.matchTemplate(main_b, template_b, method)
            result_g = cv2.matchTemplate(main_g, template_g, method)
            result_r = cv2.matchTemplate(main_r, template_r, method)
            # 综合匹配结果
            result = (result_b + result_g + result_r) / 3
            
            # 确定当前尺度下的最佳匹配位置
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                if min_val < best_val:
                    best_val = min_val
                    best_match = (scale, min_loc, scaled_template.shape[1], scaled_template.shape[0])
            else:
                if max_val > best_val:
                    best_val = max_val
                    best_match = (scale, max_loc, scaled_template.shape[1], scaled_template.shape[0])
        
        # 判断是否匹配成功
        match_successful = False
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            if best_val <= threshold:
                match_successful = True
        else:
            if best_val >= threshold:
                match_successful = True
        
        if not match_successful:
            return None
        
        # 在主图像上绘制最佳匹配位置
        scale, top_left, w, h = best_match
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(main_image, top_left, bottom_right, (0, 255, 0), 2)
        
        # 显示结果
        if show:
            cv2.imshow('Multi-Scale Template Matching', main_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return top_left[0], top_left[1], w, h

    def crop_image(self, file_path, output_path, x, y, width=None, height=None, width_ratio=None, height_ratio=None):
        """
        裁剪图片并保存结果。

        参数：
        - file_path: 输入图片的路径。
        - output_path: 裁剪后图片的保存路径。
        - x: 裁剪区域的左上角x坐标。
        - y: 裁剪区域的左上角y坐标。
        - width: 裁剪区域的宽度（可选）。
        - height: 裁剪区域的高度（可选）。
        - width_ratio: 裁剪区域宽度的比例（可选）。
        - height_ratio: 裁剪区域高度的比例（可选）。
        """
        image = cv2.imread(file_path)
        if image is None: return None
        img_height, img_width = image.shape[:2]
        if width_ratio is not None: width = int(img_width * width_ratio)
        if height_ratio is not None: height = int(img_height * height_ratio)
        if width is None or height is None: raise ValueError("请提供裁剪区域的宽度和高度，或者它们的比例。")
        crop_img = image[y:y+height, x:x+width]
        cv2.imwrite(output_path, crop_img)
        return output_path
        # 显示裁剪后的图片（可选）
        # cv2.imshow('Cropped Image', crop_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    def do_ocr(self, file_path: str, simple=False) -> List:
        """
        对图像文件进行OCR识别

        参数:
        file_path (str): 图像文件路径

        返回:
        List: OCR识别结果
        """
        data = self.ocr.ocr(file_path, cls=False)[0]
        if simple: return self.get_all_text(data)
        self.data = data
        return data
    
    def search_text(self, query: str, data: List[List[Any]] = None, threshold: float = 0.6) -> List[Tuple[str, Any]]:
        """
        在OCR识别结果中搜索与query最相似的文本项，并按照相似度排序。

        参数:
        data (List[List[Any]]): OCR识别结果的数据。
        query (str): 要搜索的关键词。
        threshold (float): 相似度阈值，默认为0.6。

        返回:
        List[Tuple[str, Any]]: 按相似度排序的搜索结果列表，包含文本和相似度。
        """
        data = data if data else self.data
        results = []
        
        for item in data:
            points = [(int(x), int(y)) for x, y in item[0]]
            text, confidence = item[1]
            similarity = difflib.SequenceMatcher(None, query, text).ratio()
            if similarity >= threshold or query in text:
                results.append({'text': text, 
                                'similarity': similarity, 
                                'confidence':confidence, 
                                'position': {
                                    'p': points,
                                    'c': ((points[0][0]+points[1][0])/2, (points[0][1]+points[3][1])/2),
                                    'w': points[1][0]+points[0][0],
                                    'h': points[3][1]+points[0][1]
                                }})
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def exists_text(self, query: str, data: List[List[Any]] = None):
        return len(self.search_text(query, data=data)) > 0
    
    def filter_high_confidence(self, threshold: float = 0.9, data: List[List[Any]] = None) -> List[str]:
        """
        过滤高置信度的OCR结果

        参数:
        data (List[List[Any]]): OCR识别结果的数据。
        threshold (float): 置信度阈值，默认为0.9。

        返回:
        List[str]: 高置信度的识别结果文本列表
        """
        data = data if data else self.data
        return [item[1][0] for item in data if item[1][1] >= threshold]

    def export_to_file(self, file_path: str, file_format: str = 'json', data: List[List[Any]] = None) -> None:
        """
        将OCR结果导出到文件

        参数:
        data (List[List[Any]]): OCR识别结果的数据。
        file_path (str): 导出文件的路径。
        file_format (str): 文件格式，可以是'json'或'csv'。

        返回:
        None
        """
        data = data if data else self.data
        if file_format == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        elif file_format == 'csv':
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Coordinates', 'Text', 'Confidence'])
                for item in data:
                    writer.writerow([item[0], item[1][0], item[1][1]])
        else:
            raise ValueError("Unsupported file format. Use 'json' or 'csv'.")

    def get_all_text(self, data: List[List[Any]] = None, position=False):
        """
        返回所有文本及其位置

        参数:
        data (List[List[Any]]): OCR识别结果的数据。

        返回:
        None
        """
        data = data if data else self.data
        res = []
        for item in data:
            text = str(item[1][0])  # 确保 text 是字符串类型
            points = item[0]
            res.append((text, points) if position else text) 
        return res
              
    def display_text_positions(self, data: List[List[Any]] = None) -> None:
        """
        显示所有文本及其位置

        参数:
        data (List[List[Any]]): OCR识别结果的数据。

        返回:
        None
        """
        data = data if data else self.data
        for item in data:
            text = str(item[1][0])  # 确保 text 是字符串类型
            points = item[0]
            print(f"Text: {text}, Position: {points}")
            
    def visualize_results(self, file_path: str, show: bool = True, save_path: str = None, data: List[List[Any]] = None) -> None:
        """
        可视化OCR结果，在图像上绘制识别出的文本框

        参数:
        file_path (str): 图像文件路径。
        data (List[List[Any]]): OCR识别结果的数据。
        show (bool): 是否显示图像，默认值为True。
        save_path (str): 保存图像的路径，如果为None则不保存图像，默认值为None。

        返回:
        None
        """
        data = data if data else self.data
        # 读取图像并转换为PIL图像
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        # 使用黑体字体，可以选择合适的字体文件
        font = ImageFont.truetype("simhei.ttf", 20)  
        
        for item in data:
            points = item[0]
            # 确保 text 是字符串类型
            text = str(item[1][0])  
            # 确保 points 中的坐标是整数类型
            points = [(int(x), int(y)) for x, y in points]
            draw.polygon(points, outline=(0, 255, 0))
            draw.text((points[0][0], points[0][1] - 20), text, font=font, fill=(0, 255, 0))
        img = np.array(pil_img)
        if show:
            plt.imshow(img)
            plt.axis('off')
            plt.show()

        if save_path: pil_img.save(save_path)



def find_best_match(properties, query):
    names = [prop['name'] for prop in properties]
    best_match = process.extractOne(query, names)
    if best_match:
        best_name = best_match[0]
        for prop in properties:
            if prop['name'] == best_name:
                return prop['value']
    return None

def main():
    ocr = Ocr()
    handler = WinHandler()
    if not handler.choose_window(): return
    operator = WinOperator(handler.window)
    
    def find_and_click(text):
        search_results = ocr.search_text(query=text)
        if len(search_results) == 0: return False
        x, y = search_results[0]['position']['c']
        operator.click(x, y)
        return True
    
    
    while True:
        ocr.do_ocr(handler.capture_screenshot("screenshot.png"))
        clock_head_p = ocr.multi_scale_template_match("screenshot.png", "clock_head.png")
        btn_confirm_p = ocr.multi_scale_template_match("screenshot.png", "btn_confirm.png")
        correct_icon_p = ocr.multi_scale_template_match("screenshot.png", "correct_icon.png")
        incorrect_icon_p = ocr.multi_scale_template_match("screenshot.png", "incorrect_icon.png")
        
        if btn_confirm_p: operator.click(btn_confirm_p[0], btn_confirm_p[1])
        elif ocr.exists_text("答题"): find_and_click("答题")
        elif clock_head_p:
            y = clock_head_p[1]
            ocr.crop_image("screenshot.png", "screenshot_2.png", x=0, y=(y-10) if (y-10)>0 else 10, width_ratio=1.0, height_ratio=0.5)
            question = ''.join(ocr.do_ocr("screenshot_2.png", simple=True))
            answer = find_best_match(database.properties, question)
            print(f'{question} => {answer}')
            if answer=='对' and correct_icon_p: operator.click(correct_icon_p[0], correct_icon_p[1])
            elif answer=='错' and incorrect_icon_p: operator.click(incorrect_icon_p[0], incorrect_icon_p[1])
        
        time.sleep(1)


if __name__ == "__main__":
    main()

