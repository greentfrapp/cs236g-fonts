import re


class Point:
    def __init__(self, x, y, type):
        self._x = x
        self._y = y
        self._type = type

    def __repr__(self):
        return f"{self.x} {self.y}"

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        self._type = type

    def scale(self, scale):
        self._x = int(self._x * scale)
        self._y = int(self._y * scale)


def read_components(glif):
    """Extract components from .glif file."""
    components = []
    with open(glif, "r") as file:
        found_components = re.finditer(
            r"<component (.*?)/>", file.read()
        )
        for c in found_components:
            attributes = {attr.split("=")[0]: attr.split("=")[1].strip('"') for attr in c.group(1).split()}
            components.append(attributes)
    return components


def read_contours(glif, x_scale=1, y_scale=1, x_offset=0, y_offset=0):
    """Extract contours from .glif file."""
    contours = []
    with open(glif, "r") as file:
        found_contours = re.finditer(
            r"<contour>\s*((.|\s)*?)\s*</contour>", file.read()
        )
        for c in found_contours:
            points = []
            for line in c.group(1).split("\n"):
                x = x_scale * (int(re.search(r"x=\"(.*?)\"", line).group(1))) - x_offset
                y = y_scale * (int(re.search(r"y=\"(.*?)\"", line).group(1))) + y_offset
                try:
                    type = re.search(r"type=\"(.*?)\"", line).group(1)
                except AttributeError:
                    type = None
                point = Point(x, y, type)
                points.append(point)
            # To simplify processing, we circularly permute points
            # until certain conditions hold
            if all([p.type is None for p in points]):
                points[-1].type = "qcurve"
            while True:
                # Break if contour starts with "line" or "move"
                if points[0].type in ["move", "line"]:
                    break
                # Break if contour ends with "curve" or "qcurve"
                if points[-1].type in ["curve", "qcurve"]:
                    break
                points.append(points.pop(0))
            contours.append(points)
    return contours


def fill_oncurve_points(contours):
    """Fill in on-curve points."""
    for i, points in enumerate(contours):
        offcurves = []
        filled_points = []
        for p in points:
            if p.type is None:
                offcurves.append(p)
            elif p.type == "qcurve":
                for p_a, p_b in zip(offcurves[:-1], offcurves[1:]):
                    # An on-curve point is assumed to be in the middle
                    # of two consecutive off-curve points
                    # See https://unifiedfontobject.org/versions/ufo3/glyphs/glif/#point
                    # and https://stackoverflow.com/questions/20733790/truetype-fonts-glyph-are-made-of-quadratic-bezier-why-do-more-than-one-consecu
                    implied = Point(
                        x=(p_a.x + p_b.x) // 2,
                        y=(p_a.y + p_b.y) // 2,
                        type="implied",
                    )
                    filled_points.append(p_a)
                    filled_points.append(implied)
                if len(offcurves):
                    filled_points.append(offcurves[-1])
                filled_points.append(p)
                offcurves = []
            elif p.type == "curve":
                if len(offcurves) == 0:
                    p.type = "line"
                elif len(offcurves) == 1:
                    p.type = "qcurve"
                elif len(offcurves) != 2:
                    raise TypeError("Found curve with more than 2 offcurve points.")
                filled_points += offcurves
                filled_points.append(p)
            else:
                filled_points.append(p)
        contours[i] = filled_points
    return contours


def to_path(glif, x_scale=1, y_scale=1, x_offset=0, y_offset=0, glif_dict=None):
    """Recursively converts glif to path"""
    paths = []
    components = read_components(glif)
    for component in components:
        if component['base'] == ".ttfautohint": continue
        try:
            if glif_dict is None:
                glif_file = glif.parents[0] / f"{component['base']}.glif"
            elif component['base'] in glif_dict:
                glif_file = glif_dict.get(component['base'])
                paths += to_path(
                    glif_file,
                    x_scale=float(component.get("xScale", 1)),
                    y_scale=float(component.get("yScale", 1)),
                    x_offset=float(component.get("xOffset", 0)),
                    y_offset=float(component.get("yOffset", 0)),
                    glif_dict=glif_dict,
                )
        except FileNotFoundError:
            print(f"'{glif}' refers to unfound component '{component['base']}'")
    contours = read_contours(glif, x_scale, y_scale, x_offset, y_offset)
    paths += fill_oncurve_points(contours)
    return paths


def write_to_svg(contours, svg, size=128):
    """Write contours to SVG file."""

    if not len(contours):
        raise ValueError("Empty contours list")

    # Get min and max for calibrating coordinates
    # so we don't get negative values and also
    # for setting canvas size in SVG
    min_x = min([min([p.x for p in points]) for points in contours])
    max_x = max([max([p.x for p in points]) for points in contours])
    min_y = min([min([p.y for p in points]) for points in contours])
    max_y = max([max([p.y for p in points]) for points in contours])

    max_x += -min_x
    max_y += -min_y

    for points in contours:
        for p in points:
            p.x = p.x - min_x
            p.y = max_y - (p.y - min_y)

    scale = size / max(max_x, max_y)
    max_x = int(scale * max_x)
    max_y = int(scale * max_y)

    if max_x == 0 or max_y == 0:
        raise ValueError(f"Width/height of zero, given max_x = {max_x} and max_y = {max_y}")

    [[p.scale(scale) for p in points] for points in contours]

    segments = []
    for points in contours:
        if points[0].type in ["line", "move"]:
            d = f"M {points[0]} "
            if points[0].type == "move":
                points = points[1:]
        else:
            d = f"M {points[-1]} "
        cache = []
        for p in points:
            if p.type == "line":
                d += f"L {p} "
            else:
                cache.append(str(p))
                if p.type == "qcurve":
                    for i in range(0, len(cache), 2):
                        d += f"Q {' '.join(cache[i:i + 2])} "
                    cache = []
                elif p.type == "curve":
                    d += f"C {' '.join(cache)} "
                    cache = []
        d += "Z"
        segments.append(d)

    template = f"""<svg width="{max_x}" height="{max_y}" xmlns="http://www.w3.org/2000/svg">
    <path d="{' '.join(segments)}" stroke="transparent" fill="black"/>
</svg>
"""

    with open(svg, "w") as file:
        file.write(template)

    return ' '.join(segments)


def build_glif_dict(font):
    glif_dict = {}
    for glif in (font).glob("*.glif"):
        with open(glif, 'r') as file:
            glif_name = re.search(
                r"<glyph name=\"(.*?)\"", file.read()
            ).group(1)
        glif_dict[glif_name] = glif
    return glif_dict


def main(glif):
    if not glif.endswith(".glif"):
        ext = glif.split('.')[-1]
        raise ValueError(f"File extension is .{ext} instead of .glif")
    contours = read_contours(glif)
    contours = fill_oncurve_points(contours)
    write_to_svg(contours, glif.replace(".glif", ".svg"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert .glif files to .svg files.")
    parser.add_argument('-g', '--glif', type=str, help='Path to .glif file.', required=True)
    args = parser.parse_args()
    main(args.glif)
