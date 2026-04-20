import os
import math
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader
)
from dataset.data_loader import Heteromatch_dataloader, MegaDepthDataset
import os
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from tqdm import tqdm

class MatchingDataModule(pl.LightningDataModule):
    def __init__(self, args, train_data, config):
        super().__init__()
        self.args = args
        self.train_data = train_data
        self.config = config

        # placeholders for datasets
        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # put any dataset download / pre-checking here if needed
        pass

    def setup(self, stage=None):
        args = self.args
        train_dataset = []
        test_dataset = []

        # test sequence list
        if args.test_seq is not None:
            test_seqs = [args.test_seq]
        else:
            test_seqs = [
                "spot_outdoor_day_srt_under_bridge_1",
                "car_urban_day_penno_small_loop",
                "falcon_outdoor_day_fast_flight_1"
            ]

        if args.train_mega:
            # MegaDepth setup
            megadepth_root = os.path.join(args.mega_dir, "e_mega")
            npz_root = os.path.join(args.mega_dir, "train-data/megadepth_indices/scene_info_0.1_0.7")

            train_list = os.path.join(args.mega_dir, "train-data/megadepth_indices/trainvaltest_list/train_list.txt")
            test_list = os.path.join(args.mega_dir, "train-data/megadepth_indices/trainvaltest_list/test_list_short.txt")

            with open(train_list, "r") as f:
                train_npz = [name.split()[0] for name in f.readlines()]
            with open(test_list, "r") as f:
                test_npz = [name.split()[0] for name in f.readlines()]

            train_npz_names = [f"{n}.npz" for n in train_npz]
            test_npz_names = [f"{n}.npz" for n in test_npz]

            ignore_list = ["1589", "0183", "0331", "0101", "0189"]

            for npz_name in tqdm(train_npz_names, desc="Building MegaDepth train set"):
                npz_path = os.path.join(npz_root, npz_name)
                train_dataset.append(
                    MegaDepthDataset(
                        megadepth_root,
                        npz_path,
                        mode='train',
                        train_data=self.train_data,
                        min_overlap_score=self.config.DATASET.MIN_OVERLAP_SCORE_TRAIN,
                        normalize_event=True,
                        num_bins=self.config.NUM_BINS,
                        ignore_list=ignore_list,
                        train_res=self.config.RES
                    )
                )

            for npz_name in tqdm(test_npz_names, desc="Building MegaDepth test set"):
                npz_path = os.path.join(npz_root, npz_name)
                test_dataset.append(
                    MegaDepthDataset(
                        megadepth_root,
                        npz_path,
                        mode='test',
                        train_data=self.train_data,
                        min_overlap_score=self.config.DATASET.MIN_OVERLAP_SCORE_TEST,
                        normalize_event=True,
                        num_bins=self.config.NUM_BINS,
                        ignore_list=ignore_list,
                        train_res=self.config.RES
                    )
                )

        else:
            # Heteromatch loader
            loader = Heteromatch_dataloader(
                dst_res=self.config.RES,
                root_dir=args.root_dir,
                train_data=self.train_data,
                mvsec=False,
                test_set=test_seqs,
            )
            train_dataset.append(loader.get_train_dataset())
            test_dataset.append(loader.get_test_dataset())

        # Concat all datasets
        self.train_dataset = ConcatDataset(train_dataset)
        self.test_dataset = ConcatDataset(test_dataset)
       

        print("Total length of train dataset:", len(self.train_dataset))
        print("Total length of test dataset:", len(self.test_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=True,
        )


