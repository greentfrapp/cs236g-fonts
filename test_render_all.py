from pathlib import Path
from glyphs import ALPHABETS
import re

import os
import torch as th
import random
import numpy as np
from PIL import Image

import pydiffvg


TEMPLATES = []
MASKS = []

def render(canvas_width, canvas_height, shapes, shape_groups, samples=2,
           seed=None):
    if seed is None:
        seed = random.randint(0, 1000000)
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width, canvas_height, samples, samples,
                  seed,   # seed
                  None,  # background image
                  *scene_args)
    return img

def test_render(all_points, all_widths,
                  canvas_size=32):
    dev = all_points.device

    all_points = 0.5*(all_points + 1.0) * canvas_size

    eps = 1e-4
    all_points = all_points + eps*th.randn_like(all_points)

    bs, num_strokes, num_pts, _ = all_points.shape
    num_segments = (num_pts - 1) // 3
    n_out = 1
    output = th.zeros(bs, n_out, canvas_size, canvas_size,
                      device=all_points.device)

    scenes = []
    for k in range(bs):
        shapes = []
        shape_groups = []
        for p in range(num_strokes):
            points = all_points[k, p].contiguous().cpu()
            # bezier
            num_ctrl_pts = th.zeros(num_segments, dtype=th.int32) + 2
            width = all_widths[k, p]#.cpu()
            color = th.ones(4)

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                stroke_width=width, is_closed=False)
            shapes.append(path)
        path_group = pydiffvg.ShapeGroup(
            shape_ids=th.tensor(list(range(len(all_points[k])))),
            fill_color=color,
            stroke_color=None)
        shape_groups.append(path_group)

        # Rasterize
        scenes.append((canvas_size, canvas_size, shapes, shape_groups))
        raster = render(canvas_size, canvas_size, shapes, shape_groups,
                        samples=2)
        raster = raster.permute(2, 0, 1).view(4, canvas_size, canvas_size)

        image = raster[:1]
        output[k] = image

    output = output.to(dev)

    return output, scenes


font_folder = Path("data_all/svg/NotoSans-Regular")

path_rgx = re.compile('<path (.*?)/>')

paths = {}
max_max = 0
for glyph in ALPHABETS:
    print(glyph)
    glyph_file = font_folder / f"{glyph}.svg"
    with open(glyph_file, "r") as file:
        found_components = re.finditer(
            r'<path d="(.*?)" .*?/>', file.read()
        )
    coords = []
    subpath = None
    max_length = 169
    max_paths = 3
    length = 0
    for i, c in enumerate(found_components):
        coord = []
        for el in c.group(1).split():
            if el in ['M', 'L', 'Q', 'Z']:
                subpath = el
                if el == 'Z':
                    for _ in range(6):
                        coords[-1].append(coords[-1][0])
                        length += 1
                    if length > max_length:
                        max_length = length
                    length = 0
            elif len(coord) in [0, 2]:
                coord = [int(el)]
            else:
                coord.append(int(el))
                if subpath == 'M':
                    coords.append([])
                    coords[-1].append(coord)
                    length += 1
                elif subpath == 'L':
                    for _ in range(6):
                        coords[-1].append(coord)
                        length += 1
                elif subpath == 'Q':
                    coords[-1].append(coord)
                    length += 1
                    subpath = 'Q1'
                elif subpath == 'Q1':
                    for _ in range(5):
                        coords[-1].append(coord)
                        length += 1

    MASK = []
    for i, path in enumerate(coords):
        MASK.append([[1, 1]] * len(path))
        while len(path) < max_length:
            coords[i].append(path[-1])
            MASK[-1].append([0, 0])
    while len(coords) < max_paths:
        coords.append([[64, 64]] * max_length)
        MASK.append([[0, 0]] * max_length)
    MASKS.append(MASK)

    all_points = th.tensor([coords])
    if max_max < all_points.shape[2]:
        max_max = all_points.shape[2]
    all_points = all_points / th.tensor([[[128, 128]]]) * 2 - 1

    TEMPLATES.append(all_points.detach().numpy().tolist()[0])

    all_widths = th.tensor([[10] * len(coords)])
    canvas_size = 512
    output, scenes = test_render(
        all_points,
        all_widths,
        canvas_size,
    )
    output = output * 2.0 - 1.0
    output = output.squeeze()
    img = Image.fromarray(((output.detach().numpy() / 2) + 0.5) * 255).convert('RGB')
    img.save(f"test_render/img_{glyph}.jpg")

print(np.array(TEMPLATES).shape)
print(np.array(MASKS).shape)

print(max_max)
with open('test.txt', 'w') as file:
    file.write(str(TEMPLATES))
with open('mask.txt', 'w') as file:
    file.write(str(MASKS))



