import torch
import numpy as np
import os,cv2,time,torch,random,pytorchvideo,warnings,argparse,math
warnings.filterwarnings("ignore",category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
#from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO
from byte_track.tracker.byte_tracker import BYTETracker
from slowfast.utils.parser import load_config, parse_args
from slowfast.models.build import build_model
from byte_track.data.data_augment import preproc
from byte_track.exp import get_exp
from byte_track.utils import fuse_model, get_model_info, postprocess
from byte_track.utils.visualize import plot_tracking



import os.path as osp
class MyVideoCapture:
    
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.idx = -1
        self.end = False
        self.stack = []
        
    def read(self):
        self.idx += 1
        ret, img = self.cap.read()
        if ret:
            self.stack.append(img)
        else:
            self.end = True
        return ret, img
    
    def to_tensor(self, img):
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img.unsqueeze(0)
        
    def get_video_clip(self):
        assert len(self.stack) > 0, "clip length must large than 0 !"
        self.stack = [self.to_tensor(img) for img in self.stack]
        clip = torch.cat(self.stack).permute(-1, 0, 1, 2)
        del self.stack
        self.stack = []
        return clip
    
    def release(self):
        self.cap.release()



def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img

def ava_inference_transform(
    clip, 
    boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow 
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45], 
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes

def plot_one_box(x, img, color=[100,100,100], text_info="None",
                 velocity=None, thickness=1, fontsize=0.5, fontthickness=1):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.45)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2), 
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
    return img

def deepsort_update(Tracker, pred, xywh, np_img):
    outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
    return outputs

def save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, color_map, output_video, vis=False):
    for i, (im, pred) in enumerate(zip(yolo_preds.ims, yolo_preds.pred)):
        #im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        if pred.shape[0]:
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                if int(cls) != 0:
                    ava_label = ''
                elif trackid in id_to_ava_labels.keys():
                    ava_label = id_to_ava_labels[trackid].split(' ')[0]
                else:
                    ava_label = 'Unknow'
                text = '{} {} {}'.format(int(trackid),yolo_preds.names[int(cls)],ava_label)
                color = color_map[int(cls)]
                im = plot_one_box(box,im,color,text)
        im = im.astype(np.uint8)
        output_video.write(im)
        if vis:
            im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
            cv2.imshow("demo", im)

def inference(img,model,num_classes,confthre,nmsthre,device,decoder=None):
    img_info = {"id": 0}
    if isinstance(img, str):
        img_info["file_name"] = osp.basename(img)
        img = cv2.imread(img)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    img = img.half()  # to FP16

    with torch.no_grad():

        outputs = model(img)
        if decoder is not None:
            outputs = decoder(outputs, dtype=outputs.type())
        outputs = postprocess(
            outputs, num_classes, confthre, nmsthre
        )
        #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
    return outputs

def main(config):
    args = parse_args()
    device = config.device
    imsize = config.imsize
    model = YOLO('./models/yolov8l.pt')
    model.to(device)
    model.conf = config.conf
    model.iou = config.iou
    model.max_det = 100
    if config.classes:
        model.classes = config.classes
    path_to_config = "D:\\yzx\\yolo_slowfast-master\\byte_track\\configs\\charadas\\SLOWFAST_16x8_R50.yaml"
    cfg = load_config(args,path_to_config)
    checkpoint = torch.load("D:\\yzx\\yolo_slowfast-master\\byte_track\\configs\\charadas\\checkpoint_epoch_00080.pyth")
    video_model=build_model(cfg)
    video_model.load_state_dict(checkpoint['model_state'])
    video_model.eval()


    # config.ckpt= "D:\\yzx\\yolo_slowfast-master\\byte_track\\model\\latest_ckpt.pth.tar"
    # if config.trt:
    #     device = "gpu"
    # device = torch.device("cuda" if device == "gpu" else "cpu")
    # by_model = config.get_model().to(device)
    # ckpt = torch.load(config.ckpt, map_location="cpu")
    # by_model.load_state_dict(ckpt["model"])
    # by_model.eval()






    ava_labelnames,_ = AvaLabeledVideoFramePaths.read_label_map("selfutils/output.pbtxt")
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

    vide_save_path = config.output
    video=cv2.VideoCapture(config.input)
    width,height = int(video.get(3)),int(video.get(4))
    video.release()
    outputvideo = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'mp4v'), 25, (width,height))
    print("processing...")
    
    cap = MyVideoCapture(config.input)
    id_to_ava_labels = {}
    a=time.time()
    while not cap.end:
        ret, img = cap.read()
        if not ret:
            continue


        #results = inference(img,by_model,config.num_classes,config.test_conf,config.nmsthre,device)
        results = model.track(img, persist=True)

        yolo_preds = results[0]
        yolo_preds.files = results[0].path = ["img.jpg"]
        ims = [results[0].orig_img.tolist()]
        ims = np.array(ims, dtype=np.uint8)
        yolo_preds.ims = ims

        yolo_preds.pred = [results[0].boxes.data]

        yolo_preds.xywh = [results[0].boxes.xywh]
        yolo_preds.names = results[0].names

        if len(cap.stack) == 25:
            print(f"processing {cap.idx // 25}th second clips")
            clip = cap.get_video_clip()
            if yolo_preds.pred[0].shape[0]:
                inputs, inp_boxes, _ = ava_inference_transform(clip, yolo_preds.pred[0][:, 0:4], crop_size=imsize)
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(device)
                with torch.no_grad():
                    slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                    slowfaster_preds = slowfaster_preds.cpu()
                for tid, avalabel in zip(yolo_preds.pred[0][:, 5].tolist(),
                                         np.argmax(slowfaster_preds, axis=1).tolist()):
                    id_to_ava_labels[tid] = ava_labelnames[avalabel + 1]

        save_yolopreds_tovideo(yolo_preds, id_to_ava_labels, coco_color_map, outputvideo, config.show)
    print("total cost: {:.3f} s, video length: {} s".format(time.time() - a, cap.idx / 25))

    cap.release()
    outputvideo.release()
    print('saved video to:', vide_save_path)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="C:\\Users\\zyq\\Desktop\\output_video.mp4", help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default="1.mp4", help='folder to save result imgs, can not use input folder')
    # object detect config
    parser.add_argument('--imsize', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--show', action='store_true', help='show img')

    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )


    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")


    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    config = parser.parse_args()
    
    if config.input.isdigit():
        print("using local camera.")
        config.input = int(config.input)
        
    print(config)
    main(config)
