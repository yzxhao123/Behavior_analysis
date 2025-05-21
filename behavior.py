from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)

from pytorchvideo.models.hub import slowfast_r50_detection

video_model = slowfast_r50_detection(True).eval().to(device)
with torch.no_grad():
    slowfaster_preds = video_model(inputs, inp_boxes.to(device))
    slowfaster_preds = slowfaster_preds.cpu()