import cairosvg
import defcon
import extractor
from fontTools.ufoLib.errors import UFOLibError
import numpy as np
import io
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import contextlib

from glif2svg import build_glif_dict, to_path, fill_oncurve_points, write_to_svg
from glyphs import GLYPHS

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
    f = open(path, 'r')
    if not f.read(): return None
    bs = cairosvg.svg2png(file_obj=f)
    f.close()
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
            if ufo.exists(): continue
            font = defcon.Font()
            extractor.extractUFO(ttf, font)
            font.save(ufo)
        except extractor.exceptions.ExtractorError:
            with nostdout():
                print(f"ExtractorError: Unable to read {str(ttf)}")
        except UFOLibError:
            with nostdout():
                print(f"UFOLibError: Unable to create {str(ufo)}")
    return ufos


def ufo2svg(ufo_fps, svg_fp, delete_if_error=True):
    svg_fp.mkdir(parents=True, exist_ok=True)
    svgs = []
    for ufo in tqdm(ufo_fps, file=sys.stdout):
        svg = svg_fp / ufo.stem
        svgs.append(svg)
        svg.mkdir(parents=True, exist_ok=True)
        glif_dict = build_glif_dict(ufo / "glyphs")
        # for glif in (ufo / "glyphs").glob("*.glif"):
        for glyph in GLYPHS:
            glif = Path(ufo / "glyphs" / (glyph + ".glif"))
            if not glif.exists():
                with nostdout():
                    print(f"{str(glif)} not found")
                continue
            if (svg / (glif.stem + ".svg")).exists(): continue
            contours = to_path(glif, glif_dict=glif_dict)
            try:
                write_to_svg(contours, svg / (glif.stem + ".svg"))
            except ValueError as e:
                with nostdout():
                    print(f"ValueError: Unable to convert {str(glif)} with error: {e}")
                    if delete_if_error:
                        os.remove(str(glif))
                        try:
                            os.remove(str(svg / (glif.stem + ".svg")))
                        except FileNotFoundError:
                            pass
                        print(f"Deleted {str(glif)}")
    return svgs


def svg2jpg(svg_fps, jpg_fp, delete_if_error=True):
    jpg_fp.mkdir(parents=True, exist_ok=True)
    jpgs = []
    for svg in tqdm(svg_fps, file=sys.stdout):
        jpg = jpg_fp / svg.stem
        jpgs.append(jpg)
        jpg.mkdir(parents=True, exist_ok=True)
        # for glif in svg.glob("*.svg"):
        for glyph in GLYPHS:
            glif = Path(svg / (glyph + ".svg"))
            if not glif.exists():
                with nostdout():
                    print(f"{str(glif)} not found")
                continue
            if (jpg / (glif.stem + ".jpg")).exists(): continue
            try:
                img_np = svg2binmap(str(glif), size=128)
                if img_np is not None:
                    Image.fromarray(img_np * 255).save(jpg / (glif.stem + ".jpg"))
            except ValueError as e:
                with nostdout():
                    print(f"ValueError: Unable to read {str(glif)} with error: {e}")
                    if delete_if_error:
                        os.remove(str(glif))
                        try:
                            os.remove(str(jpg / (glif.stem + ".jpg")))
                        except FileNotFoundError:
                            pass
                        print(f"Deleted {str(glif)}")


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert .ttf files to .jpg files.")
    parser.add_argument('-ttf', '--ttf', type=str, help='Path to .ttf file or folder containing .ttf files.', required=True)
    args = parser.parse_args()
    main(args.ttf)
