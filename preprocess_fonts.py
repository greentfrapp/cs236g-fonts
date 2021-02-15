import cairosvg
import defcon
import extractor
from fontTools.ufoLib.errors import UFOLibError
import numpy as np
import io
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import contextlib

from glif2svg import to_path, fill_oncurve_points, write_to_svg


class DummyFile(object):
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout


def svg2binmap(path="o.svg", size=128):
    """Converts SVG to a binary image."""
    bs = cairosvg.svg2png(url=path)
    # Get alpha channel of RGBA or last channel of RGB
    img_np = np.array(Image.open(io.BytesIO(bs)).split()[-1])
    if np.max(img_np) == 0:
        img_np = np.ones_like(img_np) * 255
    # Pad to (size, size)
    height, width = img_np.shape
    img_np = np.pad(img_np, (
        (0, size - height),
        (0, size - width),
    ))
    img_np = img_np // 255  # Make values 0 or 1
    return img_np


def ttf2ufo(ttf_fps, ufo_fp):
    ufo_fp.mkdir(parents=True, exist_ok=True)
    ufos = []
    for ttf in tqdm(ttf_fps, file=sys.stdout):
        try:
            ufo = ufo_fp / (ttf.stem + ".ufo")
            ufos.append(ufo)
            font = defcon.Font()
            extractor.extractUFO(ttf, font)
            font.save(ufo)
        except extractor.ExtractorError:
            with nostdout():
                print(f"ExtractorError: Unable to read {str(ttf)}")
        except UFOLibError:
            with nostdout():
                print(f"UFOLibError: Unable to create {str(ufo)}")
    return ufos


def ufo2svg(ufo_fps, svg_fp):
    svg_fp.mkdir(parents=True, exist_ok=True)
    svgs = []
    for ufo in tqdm(ufo_fps):
        svg = svg_fp / ufo.stem
        svgs.append(svg)
        svg.mkdir(parents=True, exist_ok=True)
        for glif in (ufo / "glyphs").glob("*.glif"):
            contours = to_path(glif)
            write_to_svg(contours, svg / (glif.stem + ".svg"))
    return svgs


def svg2jpg(svg_fps, jpg_fp):
    jpg_fp.mkdir(parents=True, exist_ok=True)
    jpgs = []
    for svg in tqdm(svg_fps):
        jpg = jpg_fp / svg.stem
        jpgs.append(jpg)
        jpg.mkdir(parents=True, exist_ok=True)
        for glif in svg.glob("*.svg"):
            img_np = svg2binmap(str(glif), size=128)
            Image.fromarray(img_np * 255).save(jpg / (glif.stem + ".jpg"))


def main(ttf_fp, output="./data"):
    ttf_fp = Path(ttf_fp)
    ttfs = []
    if ttf_fp.suffix == ".ttf":
        ttfs.append(ttf_fp)
    else:
        for ttf in ttf_fp.rglob("*.ttf"):
            ttfs.append(ttf)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ufos = ttf2ufo(ttfs, output / "ufo")
    svgs = ufo2svg(ufos, output / "svg")
    svg2jpg(svgs, output / "jpg")

    """
    ttf_fp = fonts-master/ofl/abeezee/ABeeZee-Regular.ttf
    output_fp = data/ufo
    """


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert .ttf files to .jpg files.")
    parser.add_argument('-ttf', '--ttf', type=str, help='Path to .ttf file or folder containing .ttf files.', required=True)
    args = parser.parse_args()
    main(args.ttf)
