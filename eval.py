# -*- coding: utf-8 -*-

import argparse
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from config.load_config import load_yaml, DotDict

from model.craft import CRAFT
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.inference_boxes import (
    test_net,
    load_icdar2015_gt,
    load_icdar2013_gt,
    load_synthtext_gt,
    load_custom_gt
)
from utils.util import copyStateDict

def save_result_custom(img_file, img, pre_output, pre_box, gt_box, result_dir):

    img = np.array(img)
    img_copy = img.copy()
    region = pre_output[0]
    affinity = pre_output[1]

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    for i, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
        except:
            pass

    if gt_box is not None:
        for j in range(len(gt_box)):
            _gt_box = np.array(gt_box[j]["points"]).reshape(-1, 2).astype(np.int32)
            if gt_box[j]["text"] == "###":
                cv2.polylines(img, [_gt_box], True, color=(128, 128, 128), thickness=2)
            else:
                cv2.polylines(img, [_gt_box], True, color=(0, 0, 255), thickness=2)

    # draw overlay image
    overlay_img = overlay(img_copy, region, affinity, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".png"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.png"
    cv2.imwrite(overlay_image_path, overlay_img)

def save_result_synth(img_file, img, pre_output, pre_box, gt_box=None, result_dir=""):

    img = np.array(img)
    img_copy = img.copy()
    region = pre_output[0]
    affinity = pre_output[1]

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    # draw bounding boxes for prediction, color green
    for i, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
        except:
            pass

    # draw bounding boxes for gt, color red
    if gt_box is not None:
        for j in range(len(gt_box)):
            cv2.polylines(
                img,
                [np.array(gt_box[j]["points"]).astype(np.int32).reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

    # draw overlay image
    overlay_img = overlay(img_copy, region, affinity, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.jpg"
    cv2.imwrite(overlay_image_path, overlay_img)


def save_result_2015(img_file, img, pre_output, pre_box, gt_box, result_dir):

    img = np.array(img)
    img_copy = img.copy()
    region = pre_output[0]
    affinity = pre_output[1]

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    for i, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
        except:
            pass

    if gt_box is not None:
        for j in range(len(gt_box)):
            _gt_box = np.array(gt_box[j]["points"]).reshape(-1, 2).astype(np.int32)
            if gt_box[j]["text"] == "###":
                cv2.polylines(img, [_gt_box], True, color=(128, 128, 128), thickness=2)
            else:
                cv2.polylines(img, [_gt_box], True, color=(0, 0, 255), thickness=2)

    # draw overlay image
    overlay_img = overlay(img_copy, region, affinity, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.jpg"
    cv2.imwrite(overlay_image_path, overlay_img)


def save_result_2013(img_file, img, pre_output, pre_box, gt_box=None, result_dir=""):

    img = np.array(img)
    img_copy = img.copy()
    region = pre_output[0]
    affinity = pre_output[1]

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    # draw bounding boxes for prediction, color green
    for i, box in enumerate(pre_box):
        poly = np.array(box).astype(np.int32).reshape((-1))
        poly = poly.reshape(-1, 2)
        try:
            cv2.polylines(
                img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2
            )
        except:
            pass

    # draw bounding boxes for gt, color red
    if gt_box is not None:
        for j in range(len(gt_box)):
            cv2.polylines(
                img,
                [np.array(gt_box[j]["points"]).reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

    # draw overlay image
    overlay_img = overlay(img_copy, region, affinity, pre_box)

    # Save result image
    res_img_path = result_dir + "/res_" + filename + ".jpg"
    cv2.imwrite(res_img_path, img)

    overlay_image_path = result_dir + "/res_" + filename + "_box.jpg"
    cv2.imwrite(overlay_image_path, overlay_img)


def overlay(image, region, affinity, single_img_bbox):

    height, width, channel = image.shape

    region_score = cv2.resize(region, (width, height))
    affinity_score = cv2.resize(affinity, (width, height))

    overlay_region = cv2.addWeighted(image.copy(), 0.4, region_score, 0.6, 5)
    overlay_aff = cv2.addWeighted(image.copy(), 0.4, affinity_score, 0.6, 5)

    boxed_img = image.copy()
    for word_box in single_img_bbox:
        cv2.polylines(
            boxed_img,
            [word_box.astype(np.int32).reshape((-1, 1, 2))],
            True,
            color=(0, 255, 0),
            thickness=3,
        )

    temp1 = np.hstack([image, boxed_img])
    temp2 = np.hstack([overlay_region, overlay_aff])
    temp3 = np.vstack([temp1, temp2])

    return temp3


def load_test_dataset_iou(test_folder_name, config):
    config = DotDict(config)
    
    if test_folder_name == "synthtext":
        total_bboxes_gt, total_img_path = load_synthtext_gt(config.test_data_dir)

    elif test_folder_name == "icdar2013":
        total_bboxes_gt, total_img_path = load_icdar2013_gt(
            dataFolder=config.test_data_dir
        )

    elif test_folder_name == "icdar2015":
        total_bboxes_gt, total_img_path = load_icdar2015_gt(
            dataFolder=config.test_data_dir
        )

    elif test_folder_name == "custom_data":
        total_bboxes_gt, total_img_path = load_custom_gt(
            config = config, isTraining = False
        )

    else:
        print("not found test dataset")
        return None, None

    return total_bboxes_gt, total_img_path


def viz_test(img, pre_output, pre_box, gt_box, img_name, result_dir, test_folder_name):

    if test_folder_name == "synthtext":
        save_result_synth(
            img_name, img[:, :, ::-1].copy(), pre_output, pre_box, gt_box, result_dir
        )
    elif test_folder_name == "icdar2013":
        save_result_2013(
            img_name, img[:, :, ::-1].copy(), pre_output, pre_box, gt_box, result_dir
        )
    elif test_folder_name == "icdar2015":
        save_result_2015(
            img_name, img[:, :, ::-1].copy(), pre_output, pre_box, gt_box, result_dir
        )
    elif test_folder_name == "custom_data":
        save_result_custom(
            img_name, img[:, :, ::-1].copy(), pre_output, pre_box, gt_box, result_dir
        )
    else:
        print("not found test dataset")

def main_eval(model_path, backbone, config, evaluator, result_dir, buffer, model, mode):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    total_imgs_bboxes_gt, total_imgs_path = load_test_dataset_iou("custom_data", config)

    if model is None:
        if backbone == "vgg":
            model = CRAFT()
        else:
            raise Exception("Undefined architecture")

        print("Loading weights from checkpoint (" + model_path + ")")
        net_param = torch.load(model_path)
        model.load_state_dict(copyStateDict(net_param["craft"]))

        if config.cuda:
            model = model.cuda()
            cudnn.benchmark = False

    model.eval()

    total_imgs_bboxes_pre = []
    results = []

    for img_path in tqdm(total_imgs_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(config)

        with torch.no_grad():
            bboxes, polys, score_text = test_net(
                model,
                image,
                config.test.custom_data.text_threshold,
                config.test.custom_data.link_threshold,
                config.test.custom_data.low_text,
                config.test.custom_data.cuda,
                config.test.custom_data.poly,
                config.test.custom_data.canvas_size,
                config.test.custom_data.mag_ratio,
            )

        single_img_bbox = [{"points": box, "text": "###", "ignore": False} for box in bboxes]
        total_imgs_bboxes_pre.append(single_img_bbox)

        if config.test.custom_data.vis_opt:
            viz_test(
                image,
                score_text,
                pre_box=polys,
                gt_box=None,
                img_name=img_path,
                result_dir=result_dir,
                test_folder_name="custom_data"
            )

    results = [evaluator.evaluate_image(gt, pred) for gt, pred in zip(total_imgs_bboxes_gt, total_imgs_bboxes_pre)]
    metrics = evaluator.combine_results(results)
    print(metrics)
    return metrics



def cal_eval(config, data, res_dir_name, opt, mode, val_loader):
    evaluator = DetectionIoUEvaluator()
    test_config = DotDict(config.test[data])
    res_dir = os.path.join(os.path.join("exp", args.yaml), "{}".format(res_dir_name))

    if opt == "iou_eval":
        main_eval(
            config.test.trained_model,
            config.train.backbone,
            test_config,
            evaluator,
            res_dir,
            buffer=None,
            model=None,
            mode=mode,
            val_loader = val_loader
        )
    else:
        print("Undefined evaluation")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CRAFT Text Detection Eval")
    parser.add_argument(
        "--yaml",
        "--yaml_file_name",
        default="custom_data_train",
        type=str,
        help="Load configuration",
    )
    args = parser.parse_args()

    # load configure
    config = load_yaml(args.yaml)
    config = DotDict(config)

    if config["wandb_opt"]:
        wandb.init(project="evaluation", entity="gmuffiness", name=args.yaml)
        wandb.config.update(config)

    val_result_dir_name = args.yaml
    cal_eval(
        config,
        "custom_data",
        val_result_dir_name + "-custom-iou",
        opt="iou_eval",
        mode=None,
    )
