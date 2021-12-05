from abc import ABC, abstractmethod
import torchvision.transforms as T
from torch.utils.data import Dataset
def get_transforms(args):
    T_lst = []
    if args.img_size:
        T_lst.append(T.Resize((args.img_size, args.img_size)))
    T_lst.append(T.ToTensor())
    if args.normalize:
        T_lst.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return T.Compose(T_lst)
class BaseDataset(ABC, Dataset):
    def __init__(self, args, logger):
        super().__init__()
        self.args = args
        self.logger = logger
    @abstractmethod
    def path_check(self): pass

