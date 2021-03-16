import numpy as np
from pathlib import Path
from PIL import Image

from glyphs import ALPHABETS
import util


fonts = [
    # 'ABeeZee-Regular',
    # 'AdventPro-Regular',
    # 'AnnieUseYourTelescope-Regular',
    # 'BigShouldersDisplay-Regular',
    # 'LifeSavers-Regular',
    # 'LibreBaskerville-Regular',
    'BadScript-Regular',
    'Alegreya-Black',
    'Fascinate-Regular',
    'ZCOOLKuaiLe-Regular',
    'Lobster-Regular',
    'PressStart2P-Regular',
    'PlayfairDisplay-Regular',
    'Lato-Regular',
]

for font in fonts:

    font_path = Path('data/jpg') / font

    glyphs = []
    for g in ALPHABETS:
        g = np.array(Image.open(font_path / f'{g}.jpg'))
        glyphs.append([g])

    util.save_image_grid(f'{font}.jpg', np.concatenate(glyphs))