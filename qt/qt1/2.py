from PyQt5.QtWidgets import QWidget, QFileDialog, QLabel, QApplication, QMessageBox, QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import cv2
import Fromwin4
from ultralytics import YOLO
from queue import Queue, Empty
model = YOLO("/models/yolov8l.pt")


class Ui_Formwin4(QWidget, Fromwin4.Ui_Formwin4):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.is_processing = False
        self.process_thread = None

        # 按钮绑定事件
        self.pushButton_load.clicked.connect(self.openvideo)
        self.pushButton_stop.clicked.connect(self.stop_processing)
        self.pushButton_d0.clicked.connect(lambda: self.start_processing("mode_1"))  # 第一种处理方式
        self.pushButton_d1.clicked.connect(lambda: self.start_processing("mode_2"))  # 第二种处理方式
        self.pushButton_save.clicked.connect(self.save_results_to_txt)  # 保存结果为 TXT 文件

    def openvideo(self):
        """加载视频文件"""
        video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi)")
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                print("无法打开视频文件！")
                return

            # 启动定时器，开始读取视频帧
            self.timer.start(30)  # 每 30 毫秒更新一次帧

    def update_frame(self):
        """读取视频帧并更新界面"""
        ret, frame = self.cap.read()
        if ret:
            # 将原始帧显示在 label_original 上
            qt_image = self.convert_cv_qt(frame, self.label_daichuli.width(), self.label_daichuli.height())
            pixmap = QPixmap.fromImage(qt_image)
            self.label_daichuli.setPixmap(pixmap)

            # 如果正在处理，将帧放入队列
            if self.is_processing and self.process_thread:
                if not self.process_thread.frames_to_process.full():
                    self.process_thread.frames_to_process.put(frame.copy())
                else:
                    print("队列已满，丢弃旧帧！")
        else:
            # 如果读取失败，停止定时器
            self.timer.stop()
            print("视频播放结束")

    def start_processing(self, mode):
        """启动视频处理"""
        if not self.cap or not self.cap.isOpened():
            print("请先加载视频！")
            return

        if not self.is_processing:
            self.current_processing_mode = mode
            self.is_processing = True
            self.process_thread = VideoProcessingThread(parent=self, mode=mode)
            self.process_thread.processed_frame.connect(self.display_processed_frame)
            self.process_thread.result_text_updated.connect(self.update_result_text)  # 连接文本更新信号

            self.process_thread.start()
            print("开始处理视频...")

    def stop_processing(self):
        """停止视频处理"""
        if self.is_processing and self.process_thread:
            self.is_processing = False
            self.process_thread.stop()
            print("停止处理视频...")

    def display_processed_frame(self, processed_frame):
        """显示处理后的帧"""
        qt_image = self.convert_cv_qt(processed_frame, self.label_jieguo.width(), self.label_jieguo.height())
        pixmap = QPixmap.fromImage(qt_image)
        self.label_jieguo.setPixmap(pixmap)

    def update_result_text(self, result_text):
        """更新结果文本"""
        self.jieguo_2.append(result_text)  # 追加到文本框

    def save_results_to_txt(self):
        """保存结果为 TXT 文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "保存 TXT 文件", "", "TXT Files (*.txt)", options=options)
        if file_path:
            try:
                # 获取 textEdit_results 的所有内容
                all_results = self.jieguo_2.toPlainText()

                # 写入到 TXT 文件
                with open(file_path, mode="w", encoding="utf-8") as txt_file:
                    txt_file.write(all_results)

                QMessageBox.information(self, "保存成功", f"结果已保存到 {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存时发生错误：{str(e)}")

    def convert_cv_qt(self, cv_img, target_width, target_height):
        """将 OpenCV 图像 (NumPy 数组) 转换为 QImage"""
        if cv_img is None:
            return QImage()  # 如果输入为空，返回空图像

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return qt_image.scaled(target_width, target_height, Qt.KeepAspectRatio)


class VideoProcessingThread(QThread):
    """用于处理视频的后台线程"""
    processed_frame = pyqtSignal(object)  # 信号，用于传递处理后的帧
    result_text_updated = pyqtSignal(str)  # 信号，用于传递处理结果文本

    def __init__(self, parent=None, mode=None):
        super().__init__()
        self.parent = parent
        self.frames_to_process = Queue(maxsize=50)  # 缓存待处理的帧
        self.running = True
        self.mode = mode  # 当前处理模式

    def run(self):
        """逐帧处理视频"""
        while self.running:
            try:
                frame = self.frames_to_process.get(timeout=1)  # 超时时间为 1 秒
            except Empty:
                print("队列为空，等待新帧...")
                continue

            # 使用模型处理帧
            if self.mode == "mode_1":  # 目标检测模式
                results = model(frame)
            elif self.mode == "mode_2":  # 目标跟踪模式
                results = model.track(frame, persist=True)

            # 可视化检测/跟踪结果
            a_frame = results[0].plot()

            # 提取结果文本
            result_text = self.extract_results_text(results)

            # 发射信号
            self.processed_frame.emit(a_frame)
            self.result_text_updated.emit(result_text)

    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()

    def extract_results_text(self, results):
        """从 YOLO 结果中提取文本信息"""
        result_text = ""
        for box in results[0].boxes:  # 遍历检测框
            cls_id = int(box.cls)  # 类别 ID
            conf = float(box.conf)  # 置信度
            class_name = model.names[cls_id]  # 获取类别名称
            result_text += f"类别: {class_name}, 置信度: {conf:.2f}"

            # 如果是跟踪模式，添加跟踪 ID
            if hasattr(box, "id") and box.id is not None:
                result_text += f", 跟踪 ID: {int(box.id)}"
            result_text += "\n"

        return result_text


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = Ui_Formwin4()
    ui.show()
    sys.exit(app.exec_())