import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage, qRgb
import test
from PyQt5.QtWidgets import *
import sys
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd



class MainWindow(QMainWindow,test.Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.controller()

        self.display_plots()

    def controller(self):
        self.pushButton_2.clicked.connect(self.img_read)
        self.Func_1.clicked.connect(self.img_read1)
        self.Func_2.clicked.connect(self.img_read2)
        self.Func_3.clicked.connect(self.img_read3)

    def img_read(self):
        fname = QFileDialog.getOpenFileName(self, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:
            self.Dis_vid.setPixmap(QPixmap(fname[0]))
            self.Dis_vid.setWordWrap(True)
            self.Dis_vid.setScaledContents(True)

    def img_read1(self):
        fname = QFileDialog.getOpenFileName(self, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:
            self.speed_1.setPixmap(QPixmap(fname[0]))
            self.speed_1.setWordWrap(True)
            self.speed_1.setScaledContents(True)

    def img_read2(self):
        fname = QFileDialog.getOpenFileName(self, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:
            self.speed_2.setPixmap(QPixmap(fname[0]))
            self.speed_2.setWordWrap(True)
            self.speed_2.setScaledContents(True)
    def img_read3(self):
        fname = QFileDialog.getOpenFileName(self, '打开图片', './', "Images (*.png *.jpg *.bmp)")
        if fname[0]:
            self.speed_3.setPixmap(QPixmap(fname[0]))
            self.speed_3.setWordWrap(True)
            self.speed_3.setScaledContents(True)

    def display_plots(self):
        # 读取数据
        csv_file = "distance_result.csv"
        try:
            df = pd.read_csv(csv_file)

            if not {'track_id_1', 'track_id_2', 'distance'}.issubset(df.columns):
                raise ValueError("CSV文件必须包含'track_id_1', 'track_id_2', 'distance'列")

            # 对每个track_id_1找到最小距离
            min_distances = df.groupby('track_id_1')['distance'].min().reset_index()

            # 创建三个不同的图表
            fig1 = self.create_min_distance_histogram(min_distances)  # 最小距离直方图
            fig2 = self.create_all_distance_histogram(df)  # 全部距离直方图
            fig3 = self.create_box_plot(min_distances)  # 箱线图

            # 将图表显示到对应的Label上
            self.set_image_to_label(fig1, self.PD_1)
            self.set_image_to_label(fig2, self.PD_2)
            self.set_image_to_label(fig3, self.PD_3)

            # 关闭图形以释放内存
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)

        except Exception as e:
            print(f"错误: {e}")
            # 显示错误信息
            self.label1.setText(f"加载数据失败: {str(e)}")

    def create_min_distance_histogram(self, data):
        """创建最小距离直方图"""
        bins = np.arange(0, 100 + 10, 10)
        custom_xticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(data['distance'], bins=bins, color='#DAEAFA', edgecolor='#77A8C5', linewidth=1)
        ax.set_xlim(0, 110)
        ax.set_ylim(0)

        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # 设置标签和刻度
        ax.set_xlabel("Distance", fontsize=14, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=14, fontweight='bold')
        ax.set_xticks(custom_xticks)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # 加粗刻度标签
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontweight('bold')

        ax.set_title("Minimum Distance Histogram (per track)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_all_distance_histogram(self, data):
        """创建全部距离直方图(你提供的第二个直方图代码)"""
        bins = np.arange(0, 100 + 10, 10)
        custom_xticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(data['distance'], bins=bins, color='#DAEAFA', edgecolor='#77A8C5', linewidth=1)
        ax.set_xlim(0, 110)
        ax.set_ylim(0)

        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # 设置标签和刻度
        ax.set_xlabel("Distance", fontsize=14, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=14, fontweight='bold')
        ax.set_xticks(custom_xticks)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # 加粗刻度标签
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontweight('bold')

        ax.set_title("All Distances Histogram", fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_box_plot(self, data):
        """创建箱线图"""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(data['distance'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#DAEAFA', color='#77A8C5'),
                   whiskerprops=dict(color='#77A8C5'),
                   capprops=dict(color='#77A8C5'),
                   medianprops=dict(color='red'))

        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.set_ylabel("Distance", fontsize=14, fontweight='bold')
        ax.set_title("Minimum Distance Distribution Boxplot", fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)

        # 加粗刻度标签
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontweight('bold')

        plt.tight_layout()
        return fig

    def set_image_to_label(self, fig, label):
        """将matplotlib图形设置到QLabel上"""
        # 将图形保存到内存中的字节流
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)

        # 创建QPixmap并设置到Label
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        label.setPixmap(pixmap)
        label.setScaledContents(True)  # 允许图像缩放



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())



