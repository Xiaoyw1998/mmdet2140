import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
# 查看当前环境信息
# python mmdet/utils/collect_env.py
# 查看完整的config信息
# python tools/misc/print_config.py configs/detectors/detectors_htc_r50_1x_coco.py


def vis(img, dets):
    boxes = dets[:, :4].round()
    boxes = boxes.astype(np.int)
    score = dets[:, 4]
    font = cv2.FONT_HERSHEY_PLAIN
    font_size = 1
    font_thickness = 2
    for d, s in zip(boxes, score):
        cv2.rectangle(img, (d[0], d[1]), (d[2], d[3]), (0, 0, 255), 2)
        # cv2.putText(img, str(s), (d[0], d[1]), font, font_size, (255, 255, 255), font_thickness)
    return img


config_file = 'configs/detectors/detectors_htc_r50_1x_coco.py'
checkpoint_file = 'checkpoints/detectors_htc_r50_1x_coco-329b1453.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
result = inference_detector(model, 'demo/demo.jpg')
img = cv2.imread('demo/demo.jpg')
show_result_pyplot(model, img, result, score_thr=0.3)


# print(len(r[0]))  # 长为80的list，每个元素是二维numpy数组
# print(len(r[1]))  # list, 里面又有80个list， 每个元素是单个目标的mask
# conf_thres = 0.3
# for i, j in zip(r[0], r[1]):
#     index = i[:, -1] > conf_thres
#     if np.any(index) > 0:
#         i = i[index]
#         img = vis(img, i)
#         index2 = np.arange(len(j), dtype=np.int)[index].tolist()
#         for k in index2:
#             # mask = j[k][:, :, np.newaxis].repeat(3, axis=2)
#             # color = np.random.randint(0, 100, 3)
#             # img += (mask*color).astype(np.uint8)
#
#             mask = j[k]
#             img = img.astype(np.float)
#             img[:, :, 0] += mask * 50
#             img = img.astype(np.uint8)
#
# cv2.imshow('s', img)
# cv2.waitKey(0)
# print(img.max())
