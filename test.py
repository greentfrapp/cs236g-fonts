import torch
import numpy as np
from PIL import Image

from svg_models import FontGenerator

zdim = 128
n_glyphs = 52
bs = 32
gen = FontGenerator(
    num_strokes=2,
    n_glyphs=n_glyphs,
    zdim=zdim,
    imsize=128,
    n_segments=32,
)

output = gen(torch.randn(1, zdim + n_glyphs))

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(gen.parameters(), lr=0.001)

test_image = np.array(Image.open('data/jpg/ABeeZee-Italic/A_.jpg'))
test_batch = np.tile(test_image, [bs, 1, 1]).astype(np.float32)
test_batch /= 255
test_batch -= 0.5
test_batch *= 2

for i in range(1000):
    optim.zero_grad()
    output = gen(torch.randn(bs, zdim + n_glyphs))#.squeeze()
    print(output.shape)
    quit()
    loss = criterion(output, torch.tensor(test_batch))
    loss.backward()
    optim.step()
    print(loss)
    Image.fromarray(((output.detach().numpy()[0] / 2) + 0.5) * 255).convert('RGB').save(f'test/{i}.jpg')
