import math
import time
from datetime import datetime
from pathlib import Path
import PIL.Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import argparse

from models import Generator, Discriminator, ResnetGenerator_3d_conv, FontEncoder
from svg_models import FontAdjuster
from glyphs import ALPHABETS
from dataloader import get_dataloaders, FontDataset
import util


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pipeline(gan, autoencoder, dataloader):
    for batch, databatch in tqdm(enumerate(dataloader, start=1)):
        source = databatch['source'].to(device)

        emb_t = gan['gen'].encode(source)

        for i in np.arange(0, 1.05, 0.1):
            interpolation = emb_t[0] * (1 - i) + emb_t[1] * i
            interpolation = interpolation.unsqueeze(0)
            gen_output_t = gan['gen'].decode(interpolation)

            fixed_z = autoencoder['encoder_A'](gen_output_t.unsqueeze(2))
            fixed_z = autoencoder['encoder_B'](fixed_z.squeeze(2))
            z_dim_size = fixed_z.shape[1] * fixed_z.shape[2] * fixed_z.shape[3]
            fixed_z = fixed_z.view(-1, 1, z_dim_size)
            z = fixed_z.repeat(1, 52, 1)  # shape = (bs*52, z_dim)
            z = z.view(-1, z_dim_size)
            glyph_one_hot = torch.eye(52).repeat(fixed_z.shape[0], 1).to(device)  # shape = (52*bs, 52)
            z = torch.cat([z, glyph_one_hot], dim=1)
                
            gen_output_t = autoencoder['gen'](z)  # shape = (52*bs, resize, resize)
            gen_output_t = gen_output_t.view(-1, 52, 128, 128)

            util.save_image_grid(f'pipeline_{i}.jpg', gen_output_t[0, :, :, :].detach().cpu().numpy()*255)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan-dir', type=str, default=None)
    parser.add_argument('--ae-dir',  type=str, default=None)
    parser.add_argument('--start-font',  type=str, default='LibreBaskerville-Regular')
    parser.add_argument('--end-font',  type=str, default='Asap-Bold')
    args = parser.parse_args()

    # Initialize models
    gan = {
        "gen": Generator().to(device).eval(),
        "dis": Discriminator().to(device).eval(),
    }
    print(f"Loading GAN from {Path(args.gan_dir)}")
    gan["gen"].load_state_dict(torch.load(str(Path(args.gan_dir) / 'gen.ckpt'), map_location=torch.device('cpu')))
    gan["dis"].load_state_dict(torch.load(str(Path(args.gan_dir) / 'dis.ckpt'), map_location=torch.device('cpu')))

    autoencoder = {
        "encoder_A": ResnetGenerator_3d_conv(input_nc=52, output_nc=52).to(device).eval(),
        "encoder_B": FontEncoder(input_nc=52, output_nc=52).to(device).eval(),
        "gen": FontAdjuster(zdim=32*8*8).to(device).eval(),
    }
    print(f"Loading autoencoder from {Path(args.ae_dir)}")
    autoencoder["gen"].load_state_dict(torch.load(str(Path(args.ae_dir) / 'gen.ckpt'), map_location=torch.device('cpu')))
    autoencoder["encoder_A"].load_state_dict(torch.load(str(Path(args.ae_dir) / 'enc_A.ckpt'), map_location=torch.device('cpu')))
    autoencoder["encoder_B"].load_state_dict(torch.load(str(Path(args.ae_dir) / 'enc_B.ckpt'), map_location=torch.device('cpu')))

    fonts = [
        args.start_font,
        args.end_font,
    ]
    dataset = FontDataset('data/jpg', fonts=fonts)
    loader = DataLoader(dataset,
                        batch_size=len(fonts),
                        sampler=SequentialSampler(dataset))

    pipeline(gan, autoencoder, loader)


if __name__ == "__main__":
    main()
