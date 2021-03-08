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
import argparse

from models import Generator, Discriminator, ResnetGenerator_3d_conv, FontEncoder
from svg_models import FontAdjuster
from glyphs import ALPHABETS
from dataloader import get_dataloaders
import util


TRAIN_ID = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log = util.get_logger('save', 'log_train_'+TRAIN_ID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
BATCH_SIZE = 8
LR = 0.0001
EPOCH_SIZE = 1000
GEN_UPDATES = 10
DIS_UPDATES = 1


def save(gen=None, dis=None, encoder_A=None, encoder_B=None):
    save_path = Path('save') / ('models_'+TRAIN_ID)
    save_path.mkdir(parents=True, exist_ok=True)
    log.info(f'Saving models to {str(save_path)}...')
    if gen:
        torch.save(gen.state_dict(), str(save_path / 'gen.ckpt'))
    if dis:
        torch.save(dis.state_dict(), str(save_path / 'dis.ckpt'))
    if encoder_A:
        torch.save(encoder_A.state_dict(), str(save_path / 'enc_A.ckpt'))
    if encoder_B:
        torch.save(encoder_B.state_dict(), str(save_path / 'enc_B.ckpt'))


def copy(gen, encoder_A, encoder_B, train_x_loader, train_y_loader, epoch, resize=128, lr=0.001, fixed_z=None):
    gen.train()
    encoder_A.train()
    encoder_B.train()
    gen.imsize = resize
    gen_losses = []
    criterion = nn.MSELoss()
    gen_optimizer = torch.optim.Adam(list(gen.parameters()) + list(encoder_A.parameters()) + list(encoder_B.parameters()), lr=lr)
    start_time = time.time()
    cur_gen_loss = np.inf
    for batch, (batch_x, batch_y) in tqdm(enumerate(zip(train_x_loader, train_y_loader), start=1)):

        source = batch_x['source'].to(device)
        target = batch_x['target'].to(device)
        real = batch_y['target'].to(device)

        gen_optimizer.zero_grad()
        
        # Update Generator
        # if fixed_z is None:
        # fixed_z = gen.sample_z(BATCH_SIZE, device=device)
        fixed_z = encoder_A(real.unsqueeze(2))
        fixed_z = encoder_B(fixed_z.squeeze(2))
        fixed_z = fixed_z.view(-1, 1, 32*8*8)
        z = fixed_z.repeat(1, 52, 1)  # shape = (bs*52, z_dim)
        z = z.view(-1, 32*8*8)
        glyph_one_hot = torch.eye(52).repeat(fixed_z.shape[0], 1).to(device)  # shape = (52*bs, 52)
        z = torch.cat([z, glyph_one_hot], dim=1)
            
        gen_output_t = gen(z)  # shape = (52*bs, resize, resize)
        gen_output_t = gen_output_t.view(-1, 52, resize, resize)
        gen_loss = criterion(gen_output_t, real)
        gen_loss.backward()
        gen_optimizer.step()
            
        gen_losses.append(gen_loss.item())
        log_interval = 10
        display_interval = 50
        if (batch % log_interval == 0 or batch == EPOCH_SIZE):
            cur_gen_loss = np.mean(gen_losses)
            elapsed = time.time() - start_time
            log.info('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f}'.format(
                    epoch, batch, EPOCH_SIZE, LR,
                    elapsed * 1000 / log_interval,
                    cur_gen_loss))
            dis_losses = []
            gen_losses = []
            start_time = time.time()
        
        if (batch % display_interval == 0 or batch == EPOCH_SIZE):
            Path(f'images/train/{TRAIN_ID}/').mkdir(parents=True, exist_ok=True)
            for i in range(len(source)):
                # util.save_image_grid(f'images/train/{TRAIN_ID}/epoch{epoch}_source_{i}.jpg', source[i, :, :, :].detach().cpu().numpy()*255)
                # util.save_image_grid(f'images/train/{TRAIN_ID}/epoch{epoch}_target_{i}.jpg', target[i, :, :, :].detach().cpu().numpy()*255)
                util.save_image_grid(f'images/train/{TRAIN_ID}/epoch{epoch}_batch{batch}_fake_{i}.jpg', gen_output_t[i, :, :, :].detach().cpu().numpy()*255)
                # util.save_image_grid(f'images/train/{TRAIN_ID}/epoch{epoch}_batch{batch}_real_{i}.jpg', real[i, :, :, :].detach().cpu().numpy()*255)
        if cur_gen_loss < 0.01: break
    return cur_gen_loss


def eval(gen, val_loader, epoch):
    Path(f'images/eval/{TRAIN_ID}/').mkdir(parents=True, exist_ok=True)
    log.info("Evaluating...")
    gen.eval()
    gen_loss = []
    criterion = nn.BCELoss()
    for batch_x in tqdm(val_loader):
        source = batch_x['source'].to(device)
        target = batch_x['target'].to(device)
    gen_output_t = gen(source)
    gen_loss.append(criterion(gen_output_t, target).item())
    gen.train()
    for i in range(len(source)):
        util.save_image_grid(f'images/eval/{TRAIN_ID}/epoch{epoch}_source_{i}.jpg', source[i, :, :, :].detach().cpu().numpy()*255)
        util.save_image_grid(f'images/eval/{TRAIN_ID}/epoch{epoch}_target_{i}.jpg', target[i, :, :, :].detach().cpu().numpy()*255)
        util.save_image_grid(f'images/eval/{TRAIN_ID}/epoch{epoch}_fake_{i}.jpg', torch.round(gen_output_t[i, :, :, :]).detach().cpu().numpy()*255)
    return np.mean(gen_loss)


def interpolate_t(embeddings):
    factors = torch.rand(embeddings.size(0), 1, 1, 1).to(device)
    perm = np.random.choice(embeddings.size(0), embeddings.size(0), replace=False)
    return embeddings * factors + embeddings[perm] * (1 - factors)


def interpolate():
    print("Interpolating...")
    gen.eval()
    x_batch = []
    y_batch = []
    for sample_glyphs in test_fonts:
        x_sample = torch.cat([glyphs_dict[s] for s in sample_glyphs], dim=1)
        x_sample[0, 5:] = 0  # Zero out all glyphs except for first 5
        x_batch.append(x_sample)
        y_batch.append(torch.cat([glyphs_dict[s] for s in sample_glyphs], dim=1))
    x_batch_t = torch.cat(x_batch, dim=0).to(device)
    y_batch_t = torch.cat(y_batch, dim=0).to(device)
    emb_t = gen.encode(x_batch_t)
    emb_t = interpolate_t(emb_t)
    gen_output_t = gen.decode(emb_t)
    display(PIL.Image.fromarray(np.concatenate(torch.round(gen_output_t[0, :16, :, :]).detach().cpu().numpy()*255, axis=1)).convert("RGB"))
    return gen_output_t

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=str, default=None)
args = parser.parse_args()

# Get DataLoaders
train_fonts = []
with open('train52_fonts.txt', 'r') as file:
    for font in file:
        train_fonts.append(font.strip())
val_fonts = []
with open('val52_fonts.txt', 'r') as file:
    for font in file:
        val_fonts.append(font.strip())
single_fonts = []
with open('single_font2.txt', 'r') as file:
    for font in file:
        single_fonts.append(font.strip())

single_fonts = train_fonts[:100]
single_fonts = [[f]*100 for f in single_fonts]

# Initialize models
gen = FontAdjuster(zdim=32*8*8).to(device)
encoder_A = ResnetGenerator_3d_conv(input_nc=52, output_nc=52).to(device)
encoder_B = FontEncoder(input_nc=52, output_nc=52).to(device)
if args.pretrain:
    print(f"Resuming from {str(Path(args.pretrain) / 'gen.ckpt')}")
    gen.load_state_dict(torch.load(str(Path(args.pretrain) / 'gen.ckpt'), map_location=torch.device('cpu')))
# dis = Discriminator(ndf=4, n_layers=2).to(device)

do_copy = True
if do_copy:
    epoch = 1
    # fixed_z = gen.sample_z(1, device=device).repeat(BATCH_SIZE, 1)
    resize_factor = 6
    # for font in single_fonts:
    min_loss = np.inf
    train_x_loader, train_y_loader, val_loader = get_dataloaders(
        'data/jpg',
        'data/jpg',
        train_fonts,
        val_fonts,
        BATCH_SIZE,
        resize=2**resize_factor,
        logger=log
    )
    EPOCH_SIZE = len(train_x_loader)
    while True:
        gen_loss = copy(
            gen,
            encoder_A,
            encoder_B,
            train_x_loader,
            train_y_loader,
            epoch,
            resize=2**resize_factor,
            lr=LR * 0.8**(epoch-1),
            # fixed_z=fixed_z
        )
        epoch += 1
        if gen_loss < min_loss:
            save(gen=gen, encoder_A=encoder_A, encoder_B=encoder_B)
            min_loss = gen_loss
        # if gen_loss < 0.01: break
