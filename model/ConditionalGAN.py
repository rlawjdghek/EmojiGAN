from os.path import join as opj

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as T

from utils import AverageMeter
from .base_model import BaseModel
from .networks import get_scheduler, GANLoss, define_G, define_D

class ConditionalGAN(BaseModel):
    def __init__(self, args, logger):
        BaseModel.__init__(self, args, logger)
        self.G = define_G(args).cuda(args.local_rank)
        self.D = define_D(args).cuda(args.local_rank)
        self.criterion_ID = nn.L1Loss()
        self.criterion_GAN = GANLoss(args.GAN_loss_name, real_label_conf=args.real_label_conf,
                                     gene_label_conf=args.gene_label_conf).cuda(args.local_rank)
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=args.G_lr, betas=args.G_betas)
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=args.D_lr, betas=args.D_betas)
        self.scheduler_G = get_scheduler(args, optimizer=self.optimizer_G)
        self.scheduler_D = get_scheduler(args, optimizer=self.optimizer_D)
        self.scaler = GradScaler()

        self.G_train_loss = AverageMeter()
        self.D_train_loss = AverageMeter()
        self.denorm_T = T.Normalize((-1, -1, -1), (2, 2, 2))
    def reset_meters(self):
        self.G_train_loss.reset()
        self.D_train_loss.reset()
    def to_train(self):
        self.G.train()
        self.D.train()
    def to_eval(self):
        self.G.eval()
        self.D.eval()
    def set_input(self, real_img, z, embedding_v):
        self.z = z
        self.embedding_v = embedding_v
        self.real_img = real_img
    def forward_G(self):
        self.gene_img = self.G(self.z, self.embedding_v)
    def get_loss_G(self):
        self.loss_GAN = self.criterion_GAN(self.D(self.gene_img), label_is_real=True)
        #loss_G = self.loss_GAN
        self.loss_ID = self.criterion_ID(self.gene_img, self.real_img) * self.args.lambda_ID
        loss_G = self.loss_GAN + self.loss_ID
        return loss_G
    def get_loss_D(self):
        pred_real = self.D(self.real_img)
        pred_gene = self.D(self.gene_img.detach())
        loss_D_real = self.criterion_GAN(pred_real, label_is_real=True)
        loss_D_gene = self.criterion_GAN(pred_gene, label_is_real=False)
        loss_D = (loss_D_real + loss_D_gene) / 2
        return loss_D
    def train(self, iter):
        self.to_train()
        self.set_requires_grad(self.D, requires_grad=False)  # 없어도 어차피 get_loss_G에서 계산은 안하지만
        # 속도가 다르다
        with autocast():
            self.forward_G()
            self.loss_G = self.get_loss_G()
        self.optimizer_G.zero_grad()
        self.scaler.scale(self.loss_G).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()

        self.set_requires_grad(self.D, requires_grad=True)
        with autocast():
            self.loss_D = self.get_loss_D()
        self.optimizer_D.zero_grad()
        self.scaler.scale(self.loss_D).backward()
        self.scaler.step(self.optimizer_D)
        self.scaler.update()

        self.G_train_loss.update(self.loss_G.item(), self.real_img.shape[0])
        self.D_train_loss.update(self.loss_D.item(), self.real_img.shape[0])
        if iter % self.args.img_save_iter_freq <= self.args.n_save_images:
            _idx = iter % self.args.img_save_iter_freq
            self.img_save(img_idx=_idx, iter=iter)
    def validation(self):
        pass
    def model_save(self, iter=None):
        to_path_G = opj(self.args.model_save_dir, f"[G]_[train]_[iteration-{iter}_{self.args.total_iter}].pth")
        to_path_D = opj(self.args.model_save_dir, f"[D]_[train]_[iteration-{iter}_{self.args.total_iter}].pth")
        G_state_dict = {}
        G_state_dict["model"] = self.G.state_dict()
        G_state_dict["optimizer"] = self.optimizer_G.state_dict()
        G_state_dict["scheduler"] = self.scheduler_G.state_dict()
        G_state_dict["iter"] = iter
        D_state_dict = {}
        D_state_dict["model"] = self.D.state_dict()
        D_state_dict["optimizer"] = self.optimizer_D.state_dict()
        D_state_dict["scheduler"] = self.scheduler_D.state_dict()
        D_state_dict["iter"] = iter
        torch.save(G_state_dict, to_path_G)
        torch.save(D_state_dict, to_path_D)
    def img_save(self, img_idx, iter=None, epoch=None, is_train=True):
        to_path = opj(self.args.img_save_dir, f"[train]_[iteration-{iter}_{self.args.total_iter}]"
                                              f"_[{img_idx}].png")
        self._img_save(self.real_img[:self.args.n_save_row].detach(), self.gene_img[:self.args.n_save_row].detach(), to_path)
    @staticmethod
    def _img_save(real_img, gene_img, to_path):
        real_img = torchvision.utils.make_grid(real_img, nrow=1, padding=0)
        gene_img = torchvision.utils.make_grid(gene_img, nrow=1, padding=0)
        real_img = F.pad(real_img, pad=(4,4,4,4))
        gene_img = F.pad(gene_img, pad=(4,4,4,4))
        save_img = torch.cat([real_img, gene_img], dim=2)
        torchvision.utils.save_image(save_img, to_path, normalize=True)
    def load_model(self):
        G_state_dict = torch.load(self.args.G_load_path)
        D_state_dict = torch.load(self.args.D_load_path)
        self.G.load_state_dict(G_state_dict["model"])
        self.optimizer_G.load_state_dict(G_state_dict["optimizer"])
        self.scheduler_G.load_state_dict(G_state_dict["scheduler"])
        self.D.load_state_dict(D_state_dict["model"])
        self.optimizer_D.load_state_dict(D_state_dict["optimizer"])
        self.scheduler_D.load_state_dict(D_state_dict["scheduler"])
        print("model is loaded successfully!!!!")
    @staticmethod
    def test_img_save(gene_img, sent, img_save_dir):
        to_path = opj(img_save_dir, f"{sent}.jpg")
        torchvision.utils.save_image(gene_img, to_path, normalize=True)
    @staticmethod
    def pruned_layers(net, rest_ratio):
        tmps = []
        for n, conv in enumerate(net.modules()):
            if isinstance(conv, nn.Conv2d) or isinstance(conv, nn.ConvTranspose2d):
                tmp_pruned = conv.weight.data.clone()
                tmp = tmp_pruned.abs().flatten()
                tmps.append(tmp)

        tmps = torch.cat(tmps)
        num = tmps.shape[0] * rest_ratio
        top_k = torch.topk(tmps, int(num), sorted=True)
        threshold = top_k.values[-1]

        for n, conv in enumerate(net.modules()):
            if isinstance(conv, nn.Conv2d) or isinstance(conv, nn.ConvTranspose2d):
                tmp_pruned = conv.weight.data.clone()
                original_size = tmp_pruned.size()
                tmp_pruned = tmp_pruned.abs().flatten()
                tmp_pruned = tmp_pruned.ge(threshold)
                tmp_pruned = tmp_pruned.contiguous().view(original_size)  # out, ch, h, w
                prune.custom_from_mask(conv, name='weight', mask=tmp_pruned)
        return net
    def pruning(self):
        self.G = self.pruned_layers(self.G, self.args.pruning_rest_ratio)
        self.D = self.pruned_layers(self.D, self.args.pruning_rest_ratio)
        print("pruning success!!!!")
