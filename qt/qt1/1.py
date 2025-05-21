from PyQt5.QtWidgets import QWidget, QFileDialog, QLabel, QApplication, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import cv2
import Fromwin4
from ultralytics import YOLO
import picture_fun
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
model = YOLO("/models/yolov8l.pt")

class Ui_Formwin6(QWidget, Fromwin4.Ui_Formwin6):
    def __init__(self):
        super(Ui_Formwin6, self).__init__()
        self.setupUi(self)

        # 初始化定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # 初始化变量
        self.frame = None  # 当前帧
        self.cap = None    # 视频捕获对象
        self.is_processing = False  # 标志是否正在处理视频
        self.frames_to_process = []  # 缓存待处理的帧
        self.processed_frames = []  # 缓存已处理的帧
        self.current_processing_mode = None  # 当前处理模式 (None, "mode_1", "mode_2")
        self.process_thread = None  # 当前处理线程

        # 按钮信号连接
        self.pushButton_load.clicked.connect(self.openvideo)
        self.pushButton_d0.clicked.connect(lambda: self.start_processing("mode_1"))  # 第一种处理方式
        self.pushButton_d1.clicked.connect(lambda: self.start_processing("mode_2"))  # 第二种处理方式

    def openvideo(self):
        """打开视频文件"""
        fname, _ = QFileDialog.getOpenFileName(self, '打开视频', "./", "Videos (*.mp4 *.mkv *.MOV *.avi)")
        if fname:
            self.cap = cv2.VideoCapture(fname)
        else:
            print("未选择视频文件！")
            return

        if self.cap.isOpened():
            self.timer.start(30)  # 每30毫秒更新一次帧
        else:
            print("无法打开视频文件！")

    def update_frame(self):
        """读取视频帧并更新界面"""
        ret, self.frame = self.cap.read()
        if ret:
            # 将原始帧显示在 label_daichuli 上
            target_width = self.label_daichuli.width()  # 获取 label 的宽度
            target_height = self.label_daichuli.height()  # 获取 label 的高度
            qt_image = self.convert_cv_qt(self.frame, target_width, target_height)
            pixmap = QPixmap.fromImage(qt_image)
            self.label_daichuli.setPixmap(pixmap)

            # 如果正在处理视频，将当前帧加入待处理队列
            if self.is_processing:
                self.frames_to_process.append(self.frame.copy())
                print(f"待处理帧数量: {len(self.frames_to_process)}")
        else:
            # 如果读取失败，停止定时器
            self.timer.stop()
            print("视频播放结束")

    def convert_cv_qt(self, cv_img, target_width, target_height):
        """将 OpenCV 图像 (NumPy 数组) 转换为 QImage"""
        if cv_img is None:
            return QImage()  # 如果输入为空，返回空图像

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)

    def start_processing(self, mode):
        """启动视频处理线程"""
        if self.cap is None:
            QMessageBox.warning(self, "警告", "请先加载视频！")
            return

        if self.is_processing:
            # 停止当前处理
            self.stop_processing()

        # 设置当前处理模式
        self.current_processing_mode = mode
        self.frames_to_process.clear()  # 清空待处理队列
        self.processed_frames.clear()   # 清空已处理队列
        self.is_processing = True

        # 启动新的处理线程
        self.process_thread = VideoProcessingThread(parent=self, mode=mode)
        self.process_thread.processed_frame.connect(self.display_processed_frame)
        self.process_thread.finished.connect(self.on_processing_finished)
        self.process_thread.start()

        print(f"开始处理视频，模式: {mode}...")

    def stop_processing(self):
        """停止当前处理"""
        if self.process_thread and self.process_thread.isRunning():
            self.process_thread.terminate()  # 强制终止线程
            self.process_thread.wait()       # 等待线程完全停止
            print("当前处理已停止！")

        # 清空界面
        self.label_jieguo.clear()
        self.is_processing = False

    def display_processed_frame(self, processed_frame):
        """显示处理后的帧"""
        target_width = self.label_jieguo.width()
        target_height = self.label_jieguo.height()
        qt_image = self.convert_cv_qt(processed_frame, target_width, target_height)
        pixmap = QPixmap.fromImage(qt_image)
        self.label_jieguo.setPixmap(pixmap)

    def on_processing_finished(self):
        """处理完成后的回调"""
        self.is_processing = False
        print("视频处理完成！")

class VideoProcessingThread(QThread):
    """用于处理视频的后台线程"""
    processed_frame = pyqtSignal(object)  # 信号，用于传递处理后的帧

    def __init__(self, parent=None, mode=None):
        super().__init__()
        self.parent = parent  # 显式传递主线程对象
        self.mode = mode      # 当前处理模式
        self.executor = ThreadPoolExecutor(max_workers=2)  # 最大并发数为 2

    def run(self):
        """逐帧处理视频"""
        while True:
            if not self.parent or not hasattr(self.parent, "frames_to_process"):
                break

            # 从队列中获取待处理帧
            try:
                frame = self.parent.frames_to_process.get(timeout=1)  # 超时时间为 1 秒
            except Empty:
                print("队列为空，等待新帧...")
                continue

            # 根据模式调用不同的处理函数
            if self.mode == "mode_1":
                future = self.executor.submit(picture_fun.detect_img, frame, model)
            elif self.mode == "mode_2":
                future = self.executor.submit(picture_fun.track_img, frame, model)
            else:
                future = self.executor.submit(lambda f: f, frame)  # 默认不处理

            # 等待推理完成并发出信号
            processed_frame = future.result()
            if processed_frame is not None:
                self.processed_frame.emit(processed_frame)  # 发出信号
                print(f"模式 {self.mode} 处理完成一帧")
        print("后台线程处理完成！")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = Ui_Formwin4()
    ui.show()
    sys.exit(app.exec_())