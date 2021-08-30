# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn.config import cfg
from predictor import COCODemo
from maskrcnn.structures.keypoint import PersonKeypoints

import time
import torch
import pyzed.sl as sl
import numpy as np
import math


def get_humans3d(prediction, depth):
    humans_3d = []
    height, width = depth.shape[:-1]
    kps = prediction.get_field("keypoints").keypoints.numpy()

    for i in range(kps.shape[0]):
        human3d = {}
        human = kps[i, :, :]

        mean_x = 0
        mean_y = 0
        mean_z = 0
        count = 0

        for kp_idx in range(human.shape[0]):

            i = int(human[kp_idx][0])
            j = int(human[kp_idx][1])

            # Median around the kp
            search_radius = 3

            bound_i_inf = i - search_radius if (i - search_radius) > 0 else 0
            bound_j_inf = j - search_radius if (j - search_radius) > 0 else 0
            bound_i_sup = i + search_radius if (i + search_radius) > width else width
            bound_j_sup = j + search_radius if (j + search_radius) > height else height

            roi = depth[bound_j_inf:bound_j_sup, bound_i_inf:bound_i_sup]

            fx = np.nanmedian(roi[:, :, 0])
            fy = np.nanmedian(roi[:, :, 1])
            fz = np.nanmedian(roi[:, :, 2])
            kp = np.array([fx, fy, fz])

            human3d[PersonKeypoints.NAMES[kp_idx]] = kp

            if not math.isnan(fx*fy*fz):
                mean_x += fx
                mean_y += fy
                mean_z += fz
                count += 1

        if count == 0:
            count = 1

        human3d['centroid'] = np.array([mean_x/count, mean_y/count, mean_z/count])
        humans_3d.append(human3d)
    return humans_3d


def get_boxes3d(prediction, depth):
    boxes_3d = []
    height, width = depth.shape[:-1]
    boxes = prediction.bbox.numpy()

    for i in range(boxes.shape[0]):
        box = boxes[i, :]

        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

        i = int((bottom_right[0]-top_left[0]) * 0.5 + top_left[0])
        j = int((bottom_right[1]-top_left[1]) * 0.5 + top_left[1])

        # Median around the centroid
        search_radius = 5

        bound_i_inf = i - search_radius if (i - search_radius) > 0 else 0
        bound_j_inf = j - search_radius if (j - search_radius) > 0 else 0
        bound_i_sup = i + search_radius if (i + search_radius) > width else width
        bound_j_sup = j + search_radius if (j + search_radius) > height else height

        roi = depth[bound_j_inf:bound_j_sup, bound_i_inf:bound_i_sup]

        fx = np.nanmedian(roi[:, :, 0])
        fy = np.nanmedian(roi[:, :, 1])
        fz = np.nanmedian(roi[:, :, 2])
        kp = np.array([fx, fy, fz])

        boxes_3d.append(kp)
    return boxes_3d


def get_masks3d(prediction, depth):
    masks_3d = []
    masks = prediction.get_field("mask").numpy()
    np_depth_flat = np.array(depth.get_data()).flatten()

    for mask in masks:
        thresh = np.array(np.squeeze(mask[0, :, :]).astype(bool)).flatten()
        x = np_depth_flat[thresh > 0]
        object_dist = np.nanmedian(x[np.isfinite(x)])
        masks_3d.append(object_dist)

    return masks_3d


def overlay_distances(prediction, boxes_3d, image, skeletons_3d=None, masks_3d=None):
    boxes = prediction.bbox

    for idx, (box, box_3d) in enumerate(zip(boxes, boxes_3d)):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        i = int((bottom_right[0]-top_left[0]) * 0.5 + top_left[0])
        j = int((bottom_right[1]-top_left[1]) * 0.5 + top_left[1])

        if masks_3d is not None:
            dist = masks_3d[idx]
        elif skeletons_3d is not None:
            pt = skeletons_3d[idx]['centroid']
            dist = math.sqrt((pt[0]) ** 2 + (pt[1]) ** 2 + (pt[2]) ** 2)
        else:
            dist = math.sqrt((box_3d[0]) ** 2 + (box_3d[1]) ** 2 + (box_3d[2]) ** 2)
            image = cv2.circle(image, (i, j), 5, (255, 0, 0), -1)

        image = cv2.putText(image, str(str("{0:.2f}".format(round(dist, 2))) + " m"), (i, j),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="configs/caffe2/e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=256,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--svo-filename",
        help="Optional SVO input filepath",
        default=None
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    init_cap_params = sl.InitParameters()
    if args.svo_filename:
        print("Loading SVO file " + args.svo_filename)
        init_cap_params.set_from_svo_file(args.svo_filename)
        init_cap_params.svo_real_time_mode = True
    init_cap_params.camera_resolution = sl.RESOLUTION.HD720
    init_cap_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_cap_params.coordinate_units = sl.UNIT.METER
    init_cap_params.depth_stabilization = True
    init_cap_params.camera_image_flip = sl.FLIP_MODE.AUTO
    init_cap_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    cap = sl.Camera()
    if not cap.is_opened():
        print("Opening ZED Camera...")
    status = cap.open(init_cap_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    display = True
    runtime = sl.RuntimeParameters()
    left = sl.Mat()
    ptcloud = sl.Mat()
    depth_img = sl.Mat()
    depth = sl.Mat()

    res = sl.Resolution(1280, 720)

    py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object
    tracking_parameters = sl.PositionalTrackingParameters(init_pos=py_transform)
    tracking_parameters.set_as_static = True
    err = cap.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    running = True
    keep_people_only = True

    if coco_demo.cfg.MODEL.MASK_ON:
        print("Mask enabled!")
    if coco_demo.cfg.MODEL.KEYPOINT_ON:
        print("Keypoints enabled!")

    while running:
        start_time = time.time()
        err_code = cap.grab(runtime)
        if err_code != sl.ERROR_CODE.SUCCESS:
            break

        cap.retrieve_image(left, sl.VIEW.LEFT, resolution=res)
        cap.retrieve_image(depth_img, sl.VIEW.DEPTH, resolution=res)
        cap.retrieve_measure(depth, sl.MEASURE.DEPTH, resolution=res)
        cap.retrieve_measure(ptcloud, sl.MEASURE.XYZ, resolution=res)
        ptcloud_np = np.array(ptcloud.get_data())

        img = cv2.cvtColor(left.get_data(), cv2.COLOR_RGBA2RGB)
        prediction = coco_demo.select_top_predictions(coco_demo.compute_prediction(img))

        # Keep people only
        if keep_people_only:
            labels_tmp = prediction.get_field("labels")
            people_coco_label = 1
            keep = torch.nonzero(labels_tmp == people_coco_label).squeeze(1)
            prediction = prediction[keep]

        composite = img.copy()
        humans_3d = None
        masks_3d = None
        if coco_demo.show_mask_heatmaps:
            composite = coco_demo.create_mask_montage(composite, prediction)
        composite = coco_demo.overlay_boxes(composite, prediction)
        if coco_demo.cfg.MODEL.MASK_ON:
            masks_3d = get_masks3d(prediction, depth)
            composite = coco_demo.overlay_mask(composite, prediction)
        if coco_demo.cfg.MODEL.KEYPOINT_ON:
            # Extract 3D skeleton from the ZED depth
            humans_3d = get_humans3d(prediction, ptcloud_np)
            composite = coco_demo.overlay_keypoints(composite, prediction)
        if True:
            overlay_distances(prediction, get_boxes3d(prediction, ptcloud_np), composite, humans_3d, masks_3d)
            composite = coco_demo.overlay_class_names(composite, prediction)

        print(" Time: {:.2f} s".format(time.time() - start_time))

        if display:
            cv2.imshow("COCO detections", composite)
            cv2.imshow("ZED Depth", depth_img.get_data())
            key = cv2.waitKey(10)
            if key == 27:
                break  # esc to quit

if __name__ == "__main__":
    main()
