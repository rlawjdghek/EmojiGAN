import time
import argparse
import os
from os.path import join as opj

import torch

from model.ConditionalGAN import ConditionalGAN
from utils import Logger
from dataset.dataloader import get_dataloader, get_test_dataloader
def build_args():
    parser = argparse.ArgumentParser()
    #### dataset ####
    parser.add_argument("--data_root_dir", type=str, default="/home/jeonghokim/EmojiGAN/data/")
    parser.add_argument("--embedding_model", type=str, default="ko",
                        choices=["ko", "kc", "mul", "xlm"])
    parser.add_argument("--data_names", nargs="+", default=["Google"], help="choose entriprises in 'Apple', 'Facebook', 'Google', "
                                                                            "'JoyPixels', 'Samsung', 'Twitter', 'Windows'")
    parser.add_argument("--out_ch", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=12)

    #### train & test ####
    parser.add_argument("--mode", type=str, default="test_from_tokenizer", choices=["train", "test_from_pickle", "test_from_tokenizer"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=200)
    parser.add_argument("--embed_dim", type=int, default=768, help="from BERT")
    parser.add_argument("--to_embed_dim", type=int, default=200, help="to reduce dimension")
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=200000)
    parser.add_argument("--lr_scheduler", type=str, default="step",
                        choices=["linear", "step", "plateau", "cosine"])
    parser.add_argument("--steplr_step", type=int, default=5000)
    parser.add_argument("--GAN_loss_name", type=str, default="lsgan",
                        choices=["vanilla", "lsgan", "wgangp"])
    parser.add_argument("--real_label_conf", type=float, default=1.0)
    parser.add_argument("--gene_label_conf", type=float, default=0.0)
    parser.add_argument("--G_lr", type=float, default=2e-4)
    parser.add_argument("--D_lr", type=float, default=2e-4)
    parser.add_argument("--G_betas", type=tuple, default=(0.5, 0.999))
    parser.add_argument("--D_betas", type=tuple, default=(0.5, 0.999))
    parser.add_argument("--lambda_ID", type=float, default=30.0)

    #### model ####
    parser.add_argument("--ngf", type=int, default=128)
    parser.add_argument("--ndf", type=int, default=128)
    parser.add_argument("--n_upsample1", type=int, default=2)
    parser.add_argument("--n_upsample2", type=int, default=2)
    parser.add_argument("--n_bottleneck", type=int, default=0)
    parser.add_argument("--padding_type", type=str, default="zero",
                        choices=["reflect", "replicate", "zero"])
    parser.add_argument("--norm_type", type=str, default="batch",
                        choices=["batch", "instance", "none"])
    parser.add_argument("--use_dropout", type=bool, default=False)
    parser.add_argument("--G_weight_init_type", type=str, default="normal")
    parser.add_argument("--G_weight_init_gain", type=float, default=0.02)
    parser.add_argument("--D_weight_init_type", type=str, default="normal")
    parser.add_argument("--D_weight_init_gain", type=float, default=0.02)
    parser.add_argument("--D_n_layers", type=int, default=4)

    #### save & load ####
    parser.add_argument("--save_root_dir", type=str, default="/media/data1/jeonghokim/EmojiGAN/save")
    parser.add_argument("--img_save_iter_freq", type=int, default=1000)
    parser.add_argument("--n_save_images", type=int, default=4)
    parser.add_argument("--n_save_row", type=int, default=10)
    parser.add_argument("--use_model_save", type=bool, default=True)
    parser.add_argument("--model_save_iter_freq", type=int, default=1000)
    parser.add_argument("--test_pickle_path", type=str, default="./test/ko_emoji_train_1130.pkl")
    parser.add_argument("--enterprise_name", type=str, default="joypixel", choices=["google", "joypixel", "twitter", "all"])
    parser.add_argument("--G_load_path", type=str, default="./test/test_model/<enterprise_name>_token7_latent200/G_lowiter.pth")
    parser.add_argument("--D_load_path", type=str, default="./test/test_model/<enterprise_name>_token7_latent200/D_lowiter.pth")
    parser.add_argument("--test_img_save_dir", type=str, default="./test/test_images/only_joypixel_train/")
    parser.add_argument("--use_pruning", type=bool, default=False)  ####
    parser.add_argument("--pruning_rest_ratio", type=float, default=0.3)

    #### config ####
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    args.save_name = f"only_face_token_7[use model save-{args.use_model_save}]_[ngf-{args.ngf}]_[ndf-{args.ndf}]_[n_upsample1-{args.n_upsample1}]_" \
                     f"[n_upsample2-{args.n_upsample2}]_[latent dim-{args.latent_dim}]_" \
                     f"[to embed dim-{args.to_embed_dim}]"
    args.G_load_path = args.G_load_path.replace("<enterprise_name>", args.enterprise_name)
    args.D_load_path = args.D_load_path.replace("<enterprise_name>", args.enterprise_name)
    args.save_dir = opj(args.save_root_dir, args.save_name)
    args.img_save_dir = opj(args.save_dir, "save_images")
    args.model_save_dir = opj(args.save_dir, "save_models")
    os.makedirs(args.img_save_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    if args.mode == "test_from_tokenizer": os.makedirs(args.test_img_save_dir, exist_ok=True)
    args.logger_path = opj(args.save_dir, "log.txt")
    args.linearlr_epochs = args.n_epochs - args.start_epoch + 1
    return args

def train(args, logger):
    train_loader = get_dataloader(args, logger)
    args.total_iter = len(train_loader) * args.n_epochs
    model = ConditionalGAN(args, logger)
    iteration = 1
    for epoch in range(1, args.n_epochs + 1):
        model.reset_meters()
        for data in train_loader:
            img = data["img"].cuda(args.local_rank)
            embedding_v = data["embedding_v"].cuda(args.local_rank)
            z = torch.randn((embedding_v.shape[0], args.latent_dim)).cuda(args.local_rank)
            model.set_input(real_img=img, z=z, embedding_v=embedding_v)
            model.train(iteration)
            if iteration % args.model_save_iter_freq == 0 and args.use_model_save:
                model.model_save(iteration)
            iteration += 1

        model.scheduler_G.step()
        model.scheduler_D.step()
        logger.write(f"[Epoch-{epoch}]_[G loss-{model.G_train_loss.avg}]"
                     f"_[D loss-{model.D_train_loss.avg}]\n")
def test_from_pickle(args):
    test_loader = get_test_dataloader(args)
    #sents = test_loader.dataset.get_sentence()
    model = ConditionalGAN(args, logger=None)
    model.load_model()
    if args.use_pruning:
        model.pruning()
    model.to_eval()
    model.G.cpu()
    for idx, data in enumerate(test_loader):
        #sent = sents[idx]
        embedding_v = data["embedding_v"].cuda(args.local_rank)
        z = torch.randn((embedding_v.shape[0], args.latent_dim)).cuda(args.local_rank)
        model.set_input(real_img=None, z=z, embedding_v=embedding_v)
        model.forward_G()
        gene_img = model.gene_img
        model.test_img_save(gene_img, idx, args.test_img_save_dir)
        #model.test_img_save(gene_img, sent, args.test_img_save_dir)
def test_from_tokenizer(args):
    import matplotlib.pyplot as plt
    model = ConditionalGAN(args, logger=None)
    model.load_model()
    model.G.cpu()
    model.D.cpu()
    if args.use_pruning:
        model.pruning()
    model.to_eval()
    from kobert_transformers import get_kobert_model, get_tokenizer
    tokenizer = get_tokenizer()
    KOBERT = get_kobert_model()
    while True:
        print(f"기업: {args.enterprise_name}")
        input_ = input("입력 문장 (종료: q): ")
        if input_ == "q": break
        else:
            embedding_v = tokenizer(input_,
                                padding="max_length",
                                max_length=7,
                                truncation=True,
                                return_tensors="pt")
            embedding_v = KOBERT(input_ids=embedding_v["input_ids"])[1]
            z = torch.randn((1, args.latent_dim))
            model.set_input(real_img=None, z=z, embedding_v=embedding_v)
            model.forward_G()
            gene_img = model.gene_img.detach().cpu().numpy()[0].transpose(1,2,0)
            plt.imshow((gene_img + 1) / 2)
            plt.show()

if __name__ == "__main__":
    args = build_args()
    logger = Logger(args.local_rank)
    logger.open(args.logger_path)
    start_t = time.time()
    if args.mode == "train": train(args, logger)
    elif args.mode == "test_from_pickle": test_from_pickle(args)
    elif args.mode == "test_from_tokenizer": test_from_tokenizer(args)
    else: raise NotImplementedError(f"Mode {args.mode} is not Implemented!!!!")
    logger.write(f"\ntime: {time.time() - start_t}")
