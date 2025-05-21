import cv2
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

        cv2.imshow("frame",frame)
        cv2.waitKey(1)
    capture.release()###释放capture
    cv2.destroyAllWindows()

