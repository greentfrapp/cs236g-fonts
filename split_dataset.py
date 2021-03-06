from pathlib import Path
import numpy as np
from glyphs import ALPHABETS


jpgs = Path('data/jpg')
valid_fonts = []

for font in jpgs.glob('*/'):
    available_glyphs = list(font.glob('*.jpg'))
    available_glyphs = [g.stem for g in available_glyphs]
    if all([g in available_glyphs for g in ALPHABETS]):
        valid_fonts.append(font.name)

train_size = int(0.9 * len(valid_fonts))
train_idxs = np.random.choice(len(valid_fonts), train_size, replace=False)

train_fonts = []
val_fonts = []
for i, font in enumerate(valid_fonts):
    if i in train_idxs:
        train_fonts.append(font)
    else:
        val_fonts.append(font)

with open('train52_fonts.txt', 'w') as file:
    for font in train_fonts:
        file.write(font + '\n')

with open('val52_fonts.txt', 'w') as file:
    for font in val_fonts:
        file.write(font + '\n')
