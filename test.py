import torch
import numpy as np
from PIL import Image

from svg_models import FontGenerator

zdim = 128
n_glyphs = 52
bs = 16
gen = FontGenerator(
    num_strokes=2,
    n_glyphs=n_glyphs,
    zdim=zdim,
    imsize=32,
    n_segments=4,
)

z = torch.randn(bs, zdim + n_glyphs)
output = gen(torch.randn(1, zdim + n_glyphs))

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(gen.parameters(), lr=0.001)

test_image = np.array(Image.open('data/jpg/ABeeZee-Italic/A_.jpg').resize((32, 32)))
test_batch = np.tile(test_image, [bs, 1, 1]).astype(np.float32)
test_batch /= 255
test_batch -= 0.5
test_batch *= 2

for i in range(1000):
    optim.zero_grad()
    output = gen(z).squeeze()
    loss = criterion(output, torch.tensor(test_batch))
    loss.backward()
    optim.step()
    print(loss)
    if i % 10 == 0:
        gen.imsize = 512
        output = gen(z).squeeze()
        Image.fromarray(((output.detach().numpy()[0] / 2) + 0.5) * 255).convert('RGB').save(f'test3/{i}.jpg')
        gen.imsize = 32
