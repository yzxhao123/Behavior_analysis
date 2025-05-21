import cv2
from ultralytics import YOLO
model = YOLO("/models/yolov8l.pt")
VIDEO_PATH="D:/yzx/yolo_slowfast-master/demo/palace.mp4"
RESULT_PATH="result.mp4"
if __name__ == '__main__':
    capture=cv2.VideoCapture(VIDEO_PATH)
    if not capture.isOpened():
        print("Error opening video file")
        exit()
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_width=capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    while True:
        success,frame=capture.read()#读取视频每一帧
        if not success:
            print("读取帧失败")
            break

        results = model.track(frame,persist=True)
        ##可视化检测框
        a_frame = results[0].plot()
        boxes = results[0].boxes.xywh.cpu()
        track_ids=results[0].boxes.id.int().cpu().tolist()
        for box,track_id in zip(boxes,track_ids):
            x,y,w,h=box


        cv2.imshow("yolo_detect",a_frame)
        cv2.waitKey(1)


    capture.release()  ###
    # 释放capture
    cv2.destroyAllWindows()

