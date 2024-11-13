# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time
import multiprocessing as mp
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.load_config import load_yaml, DotDict
from data.dataset import SynthTextDataSet, CustomDataset
from loss.mseloss import Maploss_v2, Maploss_v3
from model.craft import CRAFT
from eval import main_eval
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.util import copyStateDict, save_parser

import torch.nn.functional as F


class Trainer(object):
    def __init__(self, config, gpu, mode):

        self.config = config
        self.gpu = gpu
        self.mode = mode
        self.net_param = self.get_load_param(gpu)

    def get_synth_loader(self):

        dataset = SynthTextDataSet(
            output_size=self.config.train.data.output_size,
            data_dir=self.config.train.synth_data_dir,
            saved_gt_dir=None,
            mean=self.config.train.data.mean,
            variance=self.config.train.data.variance,
            gauss_init_size=self.config.train.data.gauss_init_size,
            gauss_sigma=self.config.train.data.gauss_sigma,
            enlarge_region=self.config.train.data.enlarge_region,
            enlarge_affinity=self.config.train.data.enlarge_affinity,
            aug=self.config.train.data.syn_aug,
            vis_test_dir=self.config.vis_test_dir,
            vis_opt=self.config.train.data.vis_opt,
            sample=self.config.train.data.syn_sample,
        )

        syn_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.train.batch_size // self.config.train.synth_ratio,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return syn_loader

    def get_custom_dataset(self):

        custom_dataset = CustomDataset(
            output_size=self.config.train.data.output_size,
            data_dir=self.config.data_root_dir,
            saved_gt_dir=None,
            mean=self.config.train.data.mean,
            variance=self.config.train.data.variance,
            gauss_init_size=self.config.train.data.gauss_init_size,
            gauss_sigma=self.config.train.data.gauss_sigma,
            enlarge_region=self.config.train.data.enlarge_region,
            enlarge_affinity=self.config.train.data.enlarge_affinity,
            watershed_param=self.config.train.data.watershed,
            aug=self.config.train.data.custom_aug,
            vis_test_dir=self.config.vis_test_dir,
            sample=self.config.train.data.custom_sample,
            vis_opt=self.config.train.data.vis_opt,
            pseudo_vis_opt=self.config.train.data.pseudo_vis_opt,
            do_not_care_label=self.config.train.data.do_not_care_label,
        )

        return custom_dataset
    
    def get_validation_loader(self):
        val_dataset = CustomDataset(
            output_size=self.config.train.data.output_size,
            data_dir=self.config.data_root_dir,
            saved_gt_dir=None,
            mean=self.config.train.data.mean,
            variance=self.config.train.data.variance,
            gauss_init_size=self.config.train.data.gauss_init_size,
            gauss_sigma=self.config.train.data.gauss_sigma,
            enlarge_region=self.config.train.data.enlarge_region,
            enlarge_affinity=self.config.train.data.enlarge_affinity,
            watershed_param=self.config.train.data.watershed,
            aug=None,
            vis_test_dir=self.config.vis_test_dir,
            sample=self.config.test.custom_data.test_set_size,
            vis_opt=self.config.train.data.vis_opt,
            pseudo_vis_opt=self.config.train.data.pseudo_vis_opt,
            do_not_care_label=self.config.train.data.do_not_care_label,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return val_loader


    def get_load_param(self, gpu):
        if self.config.train.ckpt_path is not None:
            map_location = f"cuda:{gpu}"
            param = torch.load(self.config.train.ckpt_path, map_location=map_location)
            
            # 'craft' 키가 없으면 param 전체를 craft 키로 감싸기
            if 'craft' not in param:
                print("Warning: 'craft' key missing in checkpoint. Adding the entire state_dict under 'craft'.")
                param = {"craft": param}  # 모든 파라미터를 'craft' 키로 묶어 줌
            
            return param
        else:
            return None


    def adjust_learning_rate(self, optimizer, gamma, step, lr):
        lr = lr * (gamma ** step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return param_group["lr"]

    def get_loss(self):
        if self.config.train.loss == 2:
            criterion = Maploss_v2()
        elif self.config.train.loss == 3:
            criterion = Maploss_v3()
        else:
            raise Exception("Undefined loss")
        return criterion

    def iou_eval(self, dataset, train_step, buffer, model):
        test_config = DotDict(self.config)

        val_result_dir = os.path.join(
            self.config.results_dir, "{}/{}".format(dataset + "_iou", str(train_step))
        )

        evaluator = DetectionIoUEvaluator()

        metrics = main_eval(
            None,
            self.config.train.backbone,
            test_config,
            evaluator,
            val_result_dir,
            buffer,
            model,
            self.mode,
        )
        if self.gpu == 0 and self.config.wandb_opt:
            wandb.log(
                {
                    "{} iou Recall".format(dataset): np.round(metrics["recall"], 3),
                    "{} iou Precision".format(dataset): np.round(
                        metrics["precision"], 3
                    ),
                    "{} iou F1-score".format(dataset): np.round(metrics["hmean"], 3),
                }
            )
            
        return metrics

    def train(self, buffer_dict):

        torch.cuda.set_device(self.gpu)

        # MODEL -------------------------------------------------------------------------------------------------------#
        # SUPERVISION model
        if self.config.mode == "weak_supervision":
            if self.config.train.backbone == "vgg":
                supervision_model = CRAFT(pretrained=True, amp=self.config.train.amp)
            else:
                raise Exception("Undefined architecture")

            supervision_device = self.gpu
            if self.config.train.ckpt_path is not None:
                supervision_param = self.get_load_param(supervision_device)
                supervision_model.load_state_dict(
                    copyStateDict(supervision_param["craft"])
                )
                supervision_model = supervision_model.to(f"cuda:{supervision_device}")
            print(f"Supervision model loading on : gpu {supervision_device}")
        else:
            supervision_model, supervision_device = None, None

        # TRAIN model
        if self.config.train.backbone == "vgg":
            craft = CRAFT(pretrained=True, amp=self.config.train.amp)
        else:
            raise Exception("Undefined architecture")

        if self.config.train.ckpt_path is not None:
            craft.load_state_dict(copyStateDict(self.net_param["craft"]))

        craft = craft.cuda()
        craft = torch.nn.DataParallel(craft)

        torch.backends.cudnn.benchmark = True

        # DATASET -----------------------------------------------------------------------------------------------------#

        if self.config.train.use_synthtext:
            trn_syn_loader = self.get_synth_loader()
            batch_syn = iter(trn_syn_loader)

        if self.config.train.real_dataset == "custom":
            trn_real_dataset = self.get_custom_dataset()
        else:
            raise Exception("Undefined dataset")

        if self.config.mode == "weak_supervision":
            trn_real_dataset.update_model(supervision_model)
            trn_real_dataset.update_device(supervision_device)

        trn_real_loader = torch.utils.data.DataLoader(
            trn_real_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            drop_last=False,
            pin_memory=True,
        )

        # OPTIMIZER ---------------------------------------------------------------------------------------------------#
        optimizer = optim.Adam(
            craft.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )

        if self.config.train.ckpt_path is not None and self.config.train.st_iter != 0:
            optimizer.load_state_dict(copyStateDict(self.net_param["optimizer"]))
            self.config.train.st_iter = self.net_param["optimizer"]["state"][0]["step"]
            self.config.train.lr = self.net_param["optimizer"]["param_groups"][0]["lr"]

        # LOSS --------------------------------------------------------------------------------------------------------#
        # mixed precision
        if self.config.train.amp:
            scaler = torch.cuda.amp.GradScaler()

            if (
                    self.config.train.ckpt_path is not None
                    and self.config.train.st_iter != 0
            ):
                scaler.load_state_dict(copyStateDict(self.net_param["scaler"]))
        else:
            scaler = None

        criterion = self.get_loss()

        # TRAIN -------------------------------------------------------------------------------------------------------#
        train_step = self.config.train.st_iter
        whole_training_step = self.config.train.end_iter
        update_lr_rate_step = 0
        training_lr = self.config.train.lr
        loss_value = 0
        batch_time = 0
        start_time = time.time()

        print(
            "================================ Train start ================================"
        )
        
        best_hmean = 0.0  # 성능 개선을 추적하기 위한 변수
                
        while train_step < whole_training_step:
            for index, (images, region_scores, affinity_scores, confidence_masks) in enumerate(trn_real_loader):
                craft.train()

                # CUDA로 변환
                images = images.cuda(non_blocking=True)
                region_scores = region_scores.cuda(non_blocking=True)
                affinity_scores = affinity_scores.cuda(non_blocking=True)
                confidence_masks = confidence_masks.cuda(non_blocking=True)

                # 학습률 조정
                if train_step > 0 and train_step % self.config.train.lr_decay == 0:
                    update_lr_rate_step += 1
                    training_lr = self.adjust_learning_rate(
                        optimizer,
                        self.config.train.gamma,
                        update_lr_rate_step,
                        self.config.train.lr,
                    )
                
                if self.config.train.use_synthtext:
                    # Synth image load
                    syn_image, syn_region_label, syn_affi_label, syn_confidence_mask = next(
                        batch_syn
                    )
                    syn_image = syn_image.cuda(non_blocking=True)
                    syn_region_label = syn_region_label.cuda(non_blocking=True)
                    syn_affi_label = syn_affi_label.cuda(non_blocking=True)
                    syn_confidence_mask = syn_confidence_mask.cuda(non_blocking=True)
                    # concat syn & custom image
                    images = torch.cat((syn_image, images), 0)
                    region_image_label = torch.cat(
                        (syn_region_label, region_scores), 0
                    )
                    affinity_image_label = torch.cat((syn_affi_label, affinity_scores), 0)
                    confidence_mask_label = torch.cat(
                        (syn_confidence_mask, confidence_masks), 0
                    )
                else:
                    region_image_label = region_scores
                    affinity_image_label = affinity_scores
                    confidence_mask_label = confidence_masks

                # 학습 단계 (AMP 사용 여부에 따라 분기)
                if self.config.train.amp:
                    with torch.cuda.amp.autocast():
                        output, _ = craft(images)
                        out1 = output[:, :, :, 0]
                        out2 = output[:, :, :, 1]

                        loss = criterion(
                            region_scores, affinity_scores, out1, out2, confidence_masks,
                            self.config.train.neg_rto, self.config.train.n_min_neg
                        )

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output, _ = craft(images)
                    out1 = output[:, :, :, 0]
                    out2 = output[:, :, :, 1]
                    loss = criterion(
                        region_scores, affinity_scores, out1, out2, confidence_masks,
                        self.config.train.neg_rto
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 주기적으로 손실 및 학습 정보 출력
                loss_value += loss.item()
                batch_time += time.time() - start_time

                if train_step > 0 and train_step % 5 == 0:
                    mean_loss = loss_value / 5
                    avg_batch_time = batch_time / 5
                    loss_value = 0
                    batch_time = 0

                    print(f"{time.strftime('%Y-%m-%d:%H:%M:%S')} - Step: {train_step}/{whole_training_step}, "
                        f"LR: {training_lr:.8f}, Loss: {mean_loss:.5f}, Time: {avg_batch_time:.5f}s")

                # 검증 및 모델 저장
                if train_step % self.config.train.eval_interval == 0 and train_step != 0:
                    craft.eval()

                    # 검증을 수행하고 metrics를 받아옴
                    metrics = self.iou_eval("custom_data", train_step, buffer_dict["custom_data"], craft)

                    # 검증 결과가 있을 경우에만 모델 저장
                    if metrics is not None:
                        current_hmean = metrics.get("hmean", 0)
                        print(f"Validation Results -> Precision: {metrics['precision']:.3f}, "
                            f"Recall: {metrics['recall']:.3f}, Hmean: {current_hmean:.3f}")

                        # 성능 개선 시 모델 저장
                        if current_hmean > best_hmean:
                            best_hmean = current_hmean
                            save_param_dic = {
                                "iter": train_step,
                                "craft": craft.state_dict(),
                                "optimizer": optimizer.state_dict(),
                            }
                            save_param_path = (
                                f"{self.config.results_dir}/CRAFT_best_{train_step}_hmean_{current_hmean:.4f}.pth"
                            )

                            if self.config.train.amp:
                                save_param_dic["scaler"] = scaler.state_dict()

                            torch.save(save_param_dic, save_param_path)
                            print(f"New best model saved at {save_param_path} with hmean: {current_hmean:.4f}")

                # 주기적으로 체크포인트 저장
                if train_step % self.config.train.save_interval == 0 and train_step != 0:
                    save_param_dic = {
                        "iter": train_step,
                        "craft": craft.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_param_path = f"{self.config.results_dir}/CRAFT_checkpoint_{train_step}.pth"

                    if self.config.train.amp:
                        save_param_dic["scaler"] = scaler.state_dict()

                    torch.save(save_param_dic, save_param_path)
                    print(f"Checkpoint saved at {save_param_path}")

                train_step += 1
                if train_step >= whole_training_step:
                    break


        # 훈련 종료 후 마지막 모델 저장
        save_param_dic = {
            "iter": train_step,
            "craft": craft.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_param_path = f"{self.config.results_dir}/CRAFT_last_{train_step}_hmean_{best_hmean:.4f}.pth"

        if self.config.train.amp:
            save_param_dic["scaler"] = scaler.state_dict()

        torch.save(save_param_dic, save_param_path)
        print(f"Training complete. Final model saved at {save_param_path}")



def main():
    parser = argparse.ArgumentParser(description="CRAFT custom data train")
    parser.add_argument(
        "--yaml",
        "--yaml_file_name",
        default="custom_data_train",
        type=str,
        help="Load configuration",
    )
    parser.add_argument(
        "--port", "--use ddp port", default="2346", type=str, help="Port number"
    )

    args = parser.parse_args()

    # load configure
    exp_name = args.yaml
    config = load_yaml(args.yaml)

    print("-" * 20 + " Options " + "-" * 20)
    print(yaml.dump(config))
    print("-" * 40)

    # Make result_dir
    res_dir = os.path.join(config["results_dir"], args.yaml)
    config["results_dir"] = res_dir
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Duplicate yaml file to result_dir
    shutil.copy(
        "config/" + args.yaml + ".yaml", os.path.join(res_dir, args.yaml) + ".yaml"
    )

    if config["mode"] == "weak_supervision":
        mode = "weak_supervision"
    else:
        mode = None


    # Apply config to wandb
    if config["wandb_opt"]:
        wandb.init(project="craft-stage2", entity="user_name", name=exp_name)
        wandb.config.update(config)

    config = DotDict(config)

    # Start train
    buffer_dict = {"custom_data":None}
    trainer = Trainer(config, 0, mode)
    trainer.train(buffer_dict)

    if config["wandb_opt"]:
        wandb.finish()


if __name__ == "__main__":
    main()
