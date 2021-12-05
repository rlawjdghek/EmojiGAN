from os.path import join as opj
import os
from glob import glob
import pickle
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
from .base_dataset import BaseDataset, get_transforms

class EmojiDataset(BaseDataset):
    def __init__(self, args, logger):
        BaseDataset.__init__(self, args, logger)
        self.drd = self.args.data_root_dir
        with open(opj(self.args.data_root_dir, "ko_emoji_train_1130.pkl"), "rb") as f:
            self.embedding_df = pickle.load(f)
        self.data_names = args.data_names
        self.mode = self.args.embedding_model
        self.embedding_v_name = f"{self.mode}_token_enc"
        self.path_check()
        self.T = get_transforms(self.args)
    def path_check(self):
        for name in self.data_names:
            self.logger.write(f"# {name}: {len(glob(opj(self.drd, 'only_face', name, '*.png')))}\n")
    def __len__(self):
        return len(self.embedding_df)
    def __getitem__(self, idx):
        row = self.embedding_df.iloc[idx]
        n = idx + 1
        embedding_v = row[self.embedding_v_name]
        while True:
            data_idx = random.randint(0, len(self.data_names)-1)
            name = self.data_names[data_idx]
            img_path = opj(self.drd, "new_image", name, f"{n}.png")
            if os.path.isfile(img_path): break

        img = Image.open(img_path).convert("RGB")
        data = {"img": self.T(img), "embedding_v": torch.FloatTensor(embedding_v),
                "img_number": n, "data_name_idx": data_idx}
        return data
class EmojiDataset_test(Dataset):
    def __init__(self, args):
        super().__init__()
        self.pickle_dir = args.test_pickle_path
        with open(self.pickle_dir, "rb") as f:
            self.embedding_df = pickle.load(f)
        self.mode = args.embedding_model
        self.embedding_v_name = f"{self.mode}_token_enc"
        self.T = get_transforms(args)
    def __len__(self):
        return len(self.embedding_df)
    def __getitem__(self, idx):
        row = self.embedding_df.iloc[idx]
        embedding_v = row[self.embedding_v_name]
        data = {"embedding_v": torch.FloatTensor(embedding_v)}
        return data
    def get_sentence(self):
        return self.embedding_df["emoji"]
if __name__ == "__main__":
    import sys
    sys.path.append("/home/jeonghokim/EmojiGAN/src")
    from main import build_args
    from utils import Logger
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open("/home/jeonghokim/EmojiGAN/src/deprecated")
    dataset = EmojiDataset(args, logger)
