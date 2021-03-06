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

from models import Generator, Discriminator
from glyphs import ALPHABETS
from dataloader import get_dataloaders
import util


TRAIN_ID = time.strftime('%Y%m%d_%H%M%S_GAN', time.localtime())
log = util.get_logger('save', 'log_train_'+TRAIN_ID)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
BATCH_SIZE = 16
LR = 0.001
MAX_EPOCHS = 300


def save(gen=None, dis=None):
    save_path = Path('save') / ('models_'+TRAIN_ID)
    save_path.mkdir(parents=True, exist_ok=True)
    log.info(f'Saving models to {str(save_path)}...')
    if gen:
        torch.save(gen.state_dict(), str(save_path / 'gen.ckpt'))
    if dis:
        torch.save(dis.state_dict(), str(save_path / 'dis.ckpt'))


def train(gen, dis, train_x_loader, train_y_loader, epoch, lr=0.001):
    gen.train()
    dis.train()
    dis_losses = []
    gen_losses = []
    criterion = nn.BCELoss()
    dis_optimizer = torch.optim.Adam(dis.parameters(), lr=lr)
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=lr)
    start_time = time.time()
    for batch, (batch_x, batch_y) in tqdm(enumerate(zip(train_x_loader, train_y_loader), start=1)):

        source = batch_x['source'].to(device)
        target = batch_x['target'].to(device)
        real = batch_y['target'].to(device)

        # Update Discriminator
        dis_optimizer.zero_grad()
        emb_t = gen.encode(source)
        gen_output_t = gen.decode(emb_t)
        dis_output_fake_t = dis(gen_output_t)
        dis_output_real_t = dis(real)
        dis_loss = 0.5 * torch.mean((1 - dis_output_fake_t) ** 2) + 0.5 * torch.mean(dis_output_real_t ** 2)
        dis_loss.backward()
        dis_optimizer.step()
        
        # Update Generator
        gen_optimizer.zero_grad()
        emb_t = gen.encode(source)
        gen_output_t = gen.decode(emb_t)
        dis_output_fake_t = dis(gen_output_t)
        gen_loss = 100 * criterion(gen_output_t, target) + torch.mean(dis_output_fake_t ** 2)
        gen_loss.backward()
        gen_optimizer.step()
            
        dis_losses.append(dis_loss.item())
        gen_losses.append(gen_loss.item())
        log_interval = 50
        display_interval = 500
        if (batch % log_interval == 0 or batch == len(train_x_loader)):
            cur_dis_loss = np.mean(dis_losses)
            cur_gen_loss = np.mean(gen_losses)
            elapsed = time.time() - start_time
            log.info('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f}/{:5.2f}'.format(
                    epoch, batch, len(train_x_loader), LR,
                    elapsed * 1000 / log_interval,
                    cur_dis_loss, cur_gen_loss))
            dis_losses = []
            gen_losses = []
            start_time = time.time()
        
        if (batch % display_interval == 0 or batch == len(train_x_loader)) and epoch % 10 == 0:
            Path(f'images/train/{TRAIN_ID}/').mkdir(parents=True, exist_ok=True)
            for i in range(len(source)):
                util.save_image_grid(f'images/train/{TRAIN_ID}/epoch{epoch}_source_{i}.jpg', source[i, :, :, :].detach().cpu().numpy()*255)
                util.save_image_grid(f'images/train/{TRAIN_ID}/epoch{epoch}_target_{i}.jpg', target[i, :, :, :].detach().cpu().numpy()*255)
                util.save_image_grid(f'images/train/{TRAIN_ID}/epoch{epoch}_fake_{i}.jpg', torch.round(gen_output_t[i, :, :, :]).detach().cpu().numpy()*255)


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


def main():
    # Get DataLoaders
    train_fonts = []
    with open('train52_fonts.txt', 'r') as file:
        for font in file:
            train_fonts.append(font.strip())
    val_fonts = []
    with open('val52_fonts.txt', 'r') as file:
        for font in file:
            val_fonts.append(font.strip())
    train_x_loader, train_y_loader, val_loader = get_dataloaders('data/jpg', 'data/jpg', train_fonts, val_fonts, BATCH_SIZE, logger=log)
    # Initialize models
    gen = Generator().to(device)
    dis = Discriminator().to(device)
    epoch = 1

    min_eval_loss = np.inf
    while epoch <= MAX_EPOCHS:
        train(gen, dis, train_x_loader, train_y_loader, epoch, lr=LR)
        eval_loss = eval(gen, val_loader, epoch)
        log.info(f'Eval Pixelwise BCE Loss: {eval_loss}')
        if eval_loss < min_eval_loss:
            eval_loss = min_eval_loss
            save(gen, dis)
        epoch += 1


if __name__ == '__main__':
    main()
