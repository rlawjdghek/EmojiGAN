from torch.utils.data import DataLoader
from .custom_dataset import EmojiDataset, EmojiDataset_test

def get_dataloader(args, logger):
    dataset = EmojiDataset(args, logger)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
    return dataloader
def get_test_dataloader(args):
    dataset = EmojiDataset_test(args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True)
    return dataloader


if __name__ == "__main__":
    import sys
    sys.path.append("/home/jeonghokim/EmojiGAN/src")
    from main import build_args
    from utils import Logger
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open("/home/jeonghokim/deprecated")
    train_loader = get_dataloader(args, logger)
    for data in train_loader:
        img = data["img"]
        embedding_v = data["embedding_v"]
