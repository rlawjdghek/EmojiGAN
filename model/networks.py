import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.utils.prune as prune
from einops.layers.torch import Rearrange
import functools

def get_scheduler(args, optimizer):
    if args.lr_scheduler == "linear":
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + args.start_epoch - args.n_epochs) / float(args.linearlr_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_scheduler == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.steplr_step)
    elif args.lr_scheduler == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif args.lr_scheduler == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=0)
    else:
        raise NotImplementedError(f"learning rate scheduler {args.lr_scheduler} is not impleemented!!!"
                                  f"use ['linear', 'step', 'plateau', 'cosine']")
    return scheduler
class GANLoss(nn.Module):
    def __init__(self, loss_name, real_label_conf=1.0, gene_label_conf=0.0):
        super().__init__()
        self.register_buffer("real_label", torch.tensor(real_label_conf))
        self.register_buffer("gene_label", torch.tensor(gene_label_conf))
        self.loss_name = loss_name
        if self.loss_name == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.loss_name == "lsgan":
            self.loss = nn.MSELoss()
        elif self.loss_name == "wgangp":
            self.loss = None
        else:
            raise NotImplementedError(f"GAN loss name {self.loss_name} is not implemented!!!!")
    def get_label_tensor(self, prediction, label_is_real):
        if label_is_real:
            label_tensor = self.real_label
        else:
            label_tensor = self.gene_label
        return label_tensor.expand_as(prediction)
    def forward(self, prediction, label_is_real):
        if self.loss_name == "lsgan" or self.loss_name == "vanilla":
            label = self.get_label_tensor(prediction, label_is_real)
            loss = self.loss(prediction, label)
        elif self.loss_name == "wgangp":
            if label_is_real: loss = -prediction.mean()
            else: loss = prediction.mean()
        return loss
class Identity(nn.Module):
    def forward(self, x):
        return x
def init_weight(model, init_type="normal", init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, 0.0, init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    model.apply(init_func)
    return model
def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError(f"Normalization layer {norm_type} is not implemented!!!"
                                  f"use ['batch', 'instance', 'none']")
    return norm_layer
class ConvTBlk(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super().__init__()
        self.convt_blk = self.build_blk(in_ch, out_ch, norm_layer, use_bias)

    def build_blk(self, in_ch, out_ch, norm_layer, use_bias):
        blk = []
        blk.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias))
        blk.append(norm_layer(out_ch))
        blk.append(nn.ReLU(True))
        return nn.Sequential(*blk)
    def forward(self, x):
        return self.convt_blk(x)
class ResBlk(nn.Module):
    def __init__(self, in_ch, out_ch, padding_type, norm_layer, use_bias, use_dropout):
        super().__init__()
        self.blk = self.build_blk(in_ch, out_ch, padding_type, norm_layer, use_bias, use_dropout)
    def build_blk(self, in_ch, out_ch, padding_type, norm_layer, use_bias, use_dropout):
        blk = []
        pad = 0
        if padding_type == "reflect":
            blk.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            blk.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            pad = 1
        else:
            raise NotImplementedError(f"padding type {padding_type} is not implemented!!!!"
                                      f"use ['reflect', 'replicate', 'zero']")
        blk.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=pad, bias=use_bias))
        blk.append(norm_layer(out_ch))
        blk.append(nn.ReLU(True))
        if use_dropout:
            blk.append(nn.Dropout(0.5))

        pad = 0
        if padding_type == "reflect":
            blk.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            blk.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            pad = 1
        else:
            raise NotImplementedError(f"padding type {padding_type} is not implemented!!!!"
                                      f"use ['reflect', 'replicate', 'zero']")
        blk.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=pad, bias=use_bias))
        blk.append(norm_layer(out_ch))
        return nn.Sequential(*blk)
    def forward(self, x):
        return x + self.blk(x)
class Generator(nn.Module):
    def __init__(self,
                 latent_dim,
                 embed_dim,
                 to_embed_dim,
                 out_ch,
                 ngf,
                 img_size,
                 n_upsample1,
                 n_upsample2,
                 n_bottleneck,
                 padding_type,
                 norm_layer,
                 use_dropout):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        n_upsample = n_upsample1 + n_upsample2
        self.embed_layer = nn.Linear(embed_dim, to_embed_dim)
        self.initLinear = nn.Sequential(nn.Linear(latent_dim + to_embed_dim, (ngf * (2 ** n_upsample)) * ((img_size // (2**n_upsample)) ** 2)),
                                        Rearrange("b (c h w) -> b c h w", c=ngf * (2 ** n_upsample), h=img_size // (2**n_upsample), w=img_size // (2**n_upsample)))
        blks = []
        mult = None
        for i in range(n_upsample1):
            mult = 2 ** (n_upsample - i)
            blks.append(ConvTBlk(ngf * mult, ngf * mult // 2, norm_layer, use_bias))
        for i in range(n_bottleneck):
            blks.append(ResBlk(ngf * mult // 2, ngf * mult // 2, padding_type, norm_layer, use_bias, use_dropout))
        for i in range(n_upsample1, n_upsample):
            mult = 2 ** (n_upsample - i)
            blks.append(ConvTBlk(ngf * mult, ngf * mult // 2, norm_layer, use_bias))
        blks.append(nn.Conv2d(ngf * mult // 2, out_ch, kernel_size=3, stride=1, padding=1))
        blks.append(nn.Tanh())
        self.blks = nn.Sequential(*blks)
    def forward(self, random_z, embedding):
        embedding = self.embed_layer(embedding)
        z = torch.cat([random_z, embedding], dim=1)
        x = self.initLinear(z)
        return self.blks(x)
class Discriminator(nn.Module):
    def __init__(self, in_ch, ndf, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        blks = []
        blks.append(nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1))
        blks.append(nn.LeakyReLU(0.2, True))
        mult = 1
        for i in range(1, n_layers):
            mult_prev = mult
            mult = min(2**i, 8)
            blks.append(nn.Conv2d(ndf * mult_prev, ndf * mult, kernel_size=4, stride=2, padding=1,
                                   bias=use_bias))
            blks.append(norm_layer(ndf * mult))
            blks.append(nn.LeakyReLU(0.2, True))

        mult_prev = mult
        mult = min(2 ** n_layers, 8)
        blks.append(nn.Conv2d(ndf * mult_prev, ndf * mult, kernel_size=4, stride=1, padding=1,
                               bias=use_bias))
        blks.append(norm_layer(ndf * mult))
        blks.append(nn.LeakyReLU(0.2, True))
        blks.append(nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=1))
        self.blks = nn.Sequential(*blks)
    def forward(self, x):
        return self.blks(x)
def define_G(args):
    norm_layer = get_norm_layer(args.norm_type)
    G = Generator(latent_dim=args.latent_dim,
                  embed_dim=args.embed_dim,
                  to_embed_dim=args.to_embed_dim,
                  out_ch=args.out_ch,
                  ngf=args.ngf,
                  img_size=args.img_size,
                  n_upsample1=args.n_upsample1,
                  n_upsample2=args.n_upsample2,
                  n_bottleneck=args.n_bottleneck,
                  padding_type=args.padding_type,
                  norm_layer=norm_layer,
                  use_dropout=args.use_dropout)
    return init_weight(G, init_type=args.G_weight_init_type, init_gain=args.G_weight_init_gain)
def define_D(args):
    norm_layer = get_norm_layer(args.norm_type)
    D = Discriminator(in_ch=args.out_ch,
                      ndf=args.ndf,
                      n_layers=args.D_n_layers,
                      norm_layer=norm_layer)
    return init_weight(D, init_type=args.D_weight_init_type, init_gain=args.D_weight_init_gain)

# pruned
def pruned_layers(net, rest_ratio):
    tmps = []
    for n, conv in enumerate(net.modules()):
        if isinstance(conv, nn.Conv2d) or isinstance(conv, nn.ConvTranspose2d):
            tmp_pruned = conv.weight.data.clone()
            original_size = tmp_pruned.size() # (out, ch, h, w)
            tmp = tmp_pruned.abs().flatten()
            tmps.append(tmp)

    tmps = torch.cat(tmps)
    num = tmps.shape[0] * rest_ratio
    top_k = torch.topk(tmps, int(num), sorted=True)
    threshold = top_k.values[-1]

    for n,conv in enumerate(net.modules()):
        if isinstance(conv, nn.Conv2d) or isinstance(conv, nn.ConvTranspose2d):
            tmp_pruned = conv.weight.data.clone()
            original_size = tmp_pruned.size()
            tmp_pruned = tmp_pruned.abs().flatten()
            tmp_pruned = tmp_pruned.ge(threshold)
            tmp_pruned = tmp_pruned.contiguous().view(original_size) # out, ch, h, w
            prune.custom_from_mask(conv, name='weight', mask=tmp_pruned)

    # tmps = []
    # for n, conv in enumerate(net.modules()):
    #     if isinstance(conv, nn.ConvTranspose2d):
    #         tmp_pruned = conv.weight.data.clone()
    #         original_size = tmp_pruned.size() # (out, ch, h, w)
    #         tmp = tmp_pruned.abs().flatten()
    #         tmps.append(tmp)
    #
    # tmps = torch.cat(tmps)
    # num = tmps.shape[0]*(1 - 0.5)#sparsity 0.2
    # top_k = torch.topk(tmps, int(num), sorted=True)
    # threshold = top_k.values[-1]
    #
    # for n,conv in enumerate(net.modules()):
    #     if isinstance(conv, nn.ConvTranspose2d):
    #         tmp_pruned = conv.weight.data.clone()
    #         original_size = tmp_pruned.size()
    #         tmp_pruned = tmp_pruned.abs().flatten()
    #         tmp_pruned = tmp_pruned.ge(threshold)
    #         tmp_pruned = tmp_pruned.contiguous().view(original_size) # out, ch, h, w
    #         prune.custom_from_mask(conv, name='weight', mask=tmp_pruned)
    return net

if __name__ == "__main__":
    import sys
    sys.path.append("/home/jeonghokim/EmojiGAN/src")
    from main import build_args
    args = build_args()
    sample_z = torch.randn((4, 100))
    sample_embed = torch.randn((4, 768))
    G = define_G(args)
    D = define_D(args)
    G = pruned_layers(G)
    print(G)
