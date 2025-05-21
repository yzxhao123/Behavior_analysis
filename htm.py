import streamlit as st
import cv2
import numpy as np
#import av
import torch
import tempfile
from PIL import Image
from ultralytics import YOLO
import tempfile
import cv2
import streamlit as st
import torch
from pytorchvideo.models.hub import slowfast_r50
from slowfast.visualization.video_visualizer import VideoVisualizer
from slowfast.datasets.utils import pack_pathway_output
import numpy as np


@st.experimental_singleton
def load_model():
    #model = torch.hub.load('ultralytics/yolov5','custom',path="weights/last.pt",force_reload=True)
    model = YOLO(model=r'D:\yzx\ultralytics-main\runs\train\exp10\weights\best.pt')
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 行为识别模型加载函数
@st.cache_resource
def load_behavior_model():
    model = slowfast_r50(pretrained=True).to(device)
    model = model.eval()
    return model

def preprocess_for_behavior(frame, bbox, crop_size=224):
    x1, y1, x2, y2 = bbox
    cropped = frame[int(y1):int(y2), int(x1):int(x2)]
    if cropped.size == 0:
        return None
    resized = cv2.resize(cropped, (crop_size, crop_size))
    resized = resized.astype("float32") / 255.0
    return torch.from_numpy(resized).permute(2, 0, 1)  # HWC to CHW

demo_img = "fire.9.png"
demo_video = "Fire_Video.mp4"

st.title('Animal Behavior Tracking')
st.sidebar.title('App Mode')


app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['About App','Run on Image','Run on Video','Run on WebCam','Run For Behavior'])

if app_mode == 'About App':
    st.subheader("About")
    st.markdown("<h5>This is the Animal detection, tracking and behavior recognition App</h5>",unsafe_allow_html=True)
    st.image("Images/00006.jpg")
    st.markdown("- <h5>Select the App Mode in the SideBar</h5>",unsafe_allow_html=True)
    st.markdown("- <h5>Upload the Image and Detect the animals in Images</h5>",unsafe_allow_html=True)
    st.markdown("- <h5>Upload the Video and Detect the animals in Videos</h5>",unsafe_allow_html=True)
    st.markdown("- <h5>Live Detection</h5>",unsafe_allow_html=True)
    st.markdown("- <h5>Click Start to start the camera</h5>",unsafe_allow_html=True)
    st.markdown("- <h5>Click Stop to stop the camera</h5>",unsafe_allow_html=True)

    st.markdown("""
                ## Features
- Detect on Image
- Detect on Videos
- Live Detection
- Tracking 
- Behavior Recognition
## Tech Stack
- Python
- PyTorch
- Python CV
- Streamlit
- Yolo
## Reference
[![twitter](https://img.shields.io/badge/Github-1DA1F2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AntroSafin)
""")


if app_mode == 'Run on Image':
    st.subheader("Detected Animal:")
    text = st.markdown("")

    st.sidebar.markdown("---")
    # Input for Image
    img_file = st.sidebar.file_uploader("Upload an Image",type=["jpg","jpeg","png"])
    if img_file:
        image = np.array(Image.open(img_file))
    else:
        image = np.array(Image.open(demo_img))

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Original Image**")
    st.sidebar.image(image)

    # predict the image
    model = load_model()
    results = model.predict(source=image, save=False, show=False)
    result = results[0]
    boxes = result.boxes
    length = len(boxes)
    annotated_img = result.plot()
    st.image(annotated_img, caption="Result", use_column_width=True)


if app_mode == 'Run on Video':
    st.subheader("Detected Animals:")
    text = st.markdown("")

    st.sidebar.markdown("---")

    st.subheader("Output")
    stframe = st.empty()

    #Input for Video
    video_file = st.sidebar.file_uploader("Upload a Video",type=['mp4','mov','avi','asf','m4v'])
    st.sidebar.markdown("---")
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file:
        vid = cv2.VideoCapture(demo_video)
        tffile.name = demo_video
    else:
        tffile.write(video_file.read())
        vid = cv2.VideoCapture(tffile.name)

    st.sidebar.markdown("**Input Video**")
    st.sidebar.video(tffile.name)

    # predict the video
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        model = load_model()
        results = model.predict(source=image, save=False, show=False)
        result = results[0]
        boxes = result.boxes
        length = len(boxes)
        annotated_img = result.plot()
        st.image(annotated_img, caption="Result", use_column_width=True)


if app_mode == 'Run on WebCam':
    st.subheader("Detected Fire:")
    text = st.markdown("")

    st.sidebar.markdown("---")

    st.subheader("Output")
    stframe = st.empty()

    run = st.sidebar.button("Start")
    stop = st.sidebar.button("Stop")
    st.sidebar.markdown("---")

    cam = cv2.VideoCapture(0)
    if(run):
        while(True):
            if(stop):
                break
            ret,frame = cam.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            model = load_model()
            results = model(frame)
            length = len(results.xyxy[0])
            output = np.squeeze(results.render())
            text.write(f"<h1 style='text-align: center; color:red;'>{length}</h1>",unsafe_allow_html = True)
            stframe.image(output)




