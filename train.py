import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import torch

# Importing from local modules
from tools import write2csv, setup_paths, setup_seed, log_metrics, Logger
from dataset import get_data
from method import HKD_Trainer

setup_seed(116)


def train(args):

    # ==============================
    # Configurations
    # ==============================

    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_fig = args.save_fig

    # ==============================
    # Paths
    # ==============================

    model_name, image_dir, csv_path, log_path, ckp_path, tensorboard_logger = setup_paths(args)

    logger = Logger(log_path)

    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')

    logger.info('Model name: {:}'.format(model_name))

    config_path = os.path.join('./model_configs', f'{args.model}.json')

    # ==============================
    # Load CLIP config
    # ==============================

    with open(config_path, 'r') as f:
        model_configs = json.load(f)

    n_layers = model_configs['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]

    # ==============================
    # HKD Model
    # ==============================

    model = HKD_Trainer(
        backbone=args.model,
        feat_list=features_list,
        input_dim=model_configs['vision_cfg']['width'],
        embed_dim=model_configs['embed_dim'],

        teacher_model=args.teacher_model,
        distill_weight=args.distill_weight,
        idag_weight=args.idag_weight,

        learning_rate=learning_rate,
        device=device,
        image_size=image_size

    ).to(device)

    # ==============================
    # Dataset
    # ==============================

    train_data_cls_names, train_data, train_data_root = get_data(
        dataset_type_list=args.training_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=True
    )

    test_data_cls_names, test_data, test_data_root = get_data(
        dataset_type_list=args.testing_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=False
    )

    logger.info(
        'Data Root: training, {:}; testing, {:}'.format(
            train_data_root,
            test_data_root
        )
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    # ==============================
    # Training
    # ==============================

    best_f1 = -1e10

    for epoch in tqdm(range(epochs)):

        loss_dict = model.train_epoch(train_dataloader)

        loss_total = loss_dict["total"]
        loss_kd = loss_dict["distill"]
        loss_idag = loss_dict["idag"]

        # ==============================
        # Logging
        # ==============================

        if (epoch + 1) % args.print_freq == 0:

            logger.info(
                'epoch [{}/{}], total:{:.4f}, kd:{:.4f}, idag:{:.4f}'.format(
                    epoch + 1,
                    epochs,
                    loss_total,
                    loss_kd,
                    loss_idag
                )
            )

            tensorboard_logger.add_scalar(
                'loss/total',
                loss_total,
                epoch
            )

            tensorboard_logger.add_scalar(
                'loss/kd',
                loss_kd,
                epoch
            )

            tensorboard_logger.add_scalar(
                'loss/idag',
                loss_idag,
                epoch
            )

        # ==============================
        # Validation
        # ==============================

        if (epoch + 1) % args.valid_freq == 0 or (epoch == epochs - 1):

            if epoch == epochs - 1:
                save_fig_flag = save_fig
            else:
                save_fig_flag = False

            logger.info('=============================Testing ====================================')

            metric_dict = model.evaluation(
                test_dataloader,
                test_data_cls_names,
                save_fig_flag,
                image_dir,
            )

            log_metrics(
                metric_dict,
                logger,
                tensorboard_logger,
                epoch
            )

            f1_px = metric_dict['Average']['f1_px']

            # ==============================
            # Save best model
            # ==============================

            if f1_px > best_f1:

                for k in metric_dict.keys():
                    write2csv(metric_dict[k], test_data_cls_names, k, csv_path)

                ckp_path_best = ckp_path + '_best.pth'

                model.save(ckp_path_best)

                best_f1 = f1_px


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':

    parser = argparse.ArgumentParser("HKD", add_help=True)

    # ==============================
    # Dataset
    # ==============================

    parser.add_argument(
        "--training_data",
        type=str,
        default=["mvtec", "colondb"],
        nargs='+'
    )

    parser.add_argument(
        "--testing_data",
        type=str,
        default="visa"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default='./workspaces'
    )

    # ==============================
    # CLIP backbone
    # ==============================

    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L-14-336",
        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"]
    )

    parser.add_argument(
        "--teacher_model",
        type=str,
        default="ViT-L-14"
    )

    parser.add_argument(
        "--save_fig",
        type=str2bool,
        default=False
    )

    parser.add_argument(
        "--ckt_path",
        type=str,
        default=''
    )

    # ==============================
    # Training parameters
    # ==============================

    parser.add_argument(
        "--exp_indx",
        type=int,
        default=0
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=5
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=518
    )

    parser.add_argument(
        "--print_freq",
        type=int,
        default=1
    )

    parser.add_argument(
        "--valid_freq",
        type=int,
        default=1
    )

    # ==============================
    # HKD parameters
    # ==============================

    parser.add_argument(
        "--distill_weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--idag_weight",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--k_clusters",
        type=int,
        default=20
    )

    args = parser.parse_args()

    if args.batch_size != 1:
        raise NotImplementedError(
            "Currently, only batch size of 1 is supported."
        )

    train(args)