from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from PIL import Image
from pathlib import Path
import numpy as np

import util
from glyphs import ALPHABETS


BATCH_SIZE = 16


class FontDataset(Dataset):
    def __init__(self, path, fonts=None, mask_n=42):
        self.path = Path(path)
        self.fonts = fonts
        if self.fonts is None:
            self.fonts = []
            for font in self.path.glob('*/'):
                if font.is_dir():
                    self.fonts.append(font.stem)
        self.mask_n = mask_n

    def load_font(self, font_path):
        masked = np.random.choice(52, size=self.mask_n, replace=False)
        masked_stack = []
        original_stack = []
        zeros = np.zeros((1, 128, 128), dtype=np.float32)
        for i, glyph in enumerate(ALPHABETS):
            image = np.expand_dims(np.array(Image.open(font_path / (glyph + '.jpg'))).astype(np.float32) / 255, 0)
            original_stack.append(image)
            if i in masked:
                masked_stack.append(zeros)
            else:
                masked_stack.append(image)
        return np.concatenate(masked_stack), np.concatenate(original_stack)

    def __getitem__(self, idx):
        font = self.fonts[idx]
        source, target = self.load_font(self.path / font)
        return {'source': source, 'target': target}

    def __len__(self):
        return len(self.fonts)


def get_dataloaders(train_path, val_path, train_fonts=None, val_fonts=None, batch_size=32, mask_n=42, logger=None):
    logger = logger or util.get_logger('save', 'log_train')
    logger.info("Preparing Training Data...")
    train_dataset_x = FontDataset(train_path, fonts=train_fonts, mask_n=mask_n)
    train_dataset_y = FontDataset(train_path, fonts=train_fonts, mask_n=mask_n)
    logger.info("Preparing Validation Data...")
    val_dataset = FontDataset(val_path, fonts=val_fonts, mask_n=mask_n)

    train_x_loader = DataLoader(train_dataset_x,
                            batch_size=batch_size,
                            sampler=RandomSampler(train_dataset_x))
    train_y_loader = DataLoader(train_dataset_y,
                            batch_size=batch_size,
                            sampler=RandomSampler(train_dataset_y))
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            sampler=SequentialSampler(val_dataset))
    return train_x_loader, train_y_loader, val_loader
