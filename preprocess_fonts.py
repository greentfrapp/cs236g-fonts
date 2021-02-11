import extractor
import defcon
from pathlib import Path
from PIL import Image

from glif2svg import read_contours, fill_oncurve_points, write_to_svg


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


def ttf2ufo(ttf_fp, ufo_fp):
    ttf_fp = Path(ttf_fp)
    ufo_fp = Path(ufo_fp)
    if ttf_fp.is_dir():
        ufo_fp.mkdir(parents=True, exist_ok=True)
        for ttf in ttf_fp.glob("*.ttf"):     
            ufo = defcon.Font()
            extractor.extractUFO(ttf, ufo)
            ufo.save(ufo_fp / (ttf.stem + ".ufo"))
    else:
        if not ufo_fp.endswith(".ufo"):
            ufo_fp = ufo_fp / (ttf_fp.stem + ".ufo")
        ufo = defcon.Font()
        extractor.extractUFO(ttf_fp, ufo)
        ufo.save(ufo_fp)


def ufo2svg(ufo_fp, svg_fp):
    ufo_fp = Path(ufo_fp)
    svg_fp = Path(svg_fp)
    if ufo_fp.endswith(".ufo"):
        if ufo_fp.stem not in str(svg_fp);
            svg_fp = svg_fp / ufo_fp.stem
            svg_fp.mkdir(parents=True, exist_ok=True)
        for glif in ufo_fp.glob("*.glif"):
            contours = read_contours(glif)
            contours = fill_oncurve_points(contours)
            write_to_svg(contours, svg_fp / (glif.stem + ".svg"))
    else:
        for ufo in ufo_fp.glob("*.ufo"):
            svg_fp = svg_fp / ufo_fp.stem
            svg_fp.mkdir(parents=True, exist_ok=True)
            for glif in ufo_fp.glob("*.glif"):
                contours = read_contours(glif)
                contours = fill_oncurve_points(contours)
                write_to_svg(contours, svg_fp / (glif.stem + ".svg"))


def svg2jpg(svg_fp, jpg_fp):
    svg_fp = Path(svg_fp)
    jpg_fp = Path(jpg_fp)
    if svg_fp.is_dir():
        for folder in svg_fp.glob("*/"):
            jpg_fp = jpg_fp / folder.stem
            jpg_fp.mkdir(parents=True, exist_ok=True)
            for svg in folder.glob("*.svg"):
                img_np = svg2binmap(svg, size=128)
                Image.fromarray(img_np * 255).save(jpg_fp / (svg.stem + ".jpg"))
    else:
        img_np = svg2binmap(svg_fp, size=128)
        Image.fromarray(img_np * 255).save(jpg_fp / (svg_fp.stem + ".jpg"))


def main(ttf_fp, output="./data"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ttf2ufo(ttf_fp, output / "ufo")
    ufo2svg(output / "ufo", output / "svg")
    svg2jpg(output / "svg", output / "jpg")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert .ttf files to .jpg files.")
    parser.add_argument('-ttf', '--ttf', type=str, help='Path to .ttf file or folder containing .ttf files.', required=True)
    args = parser.parse_args()
    main(args.ttf)
