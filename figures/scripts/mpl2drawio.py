# Converts a matplotlib "drawing" figure (one axes in mm coordinates, y downward) into an editable draw.io file.
# Shapes, lines, arrows and texts become native draw.io cells; raster images (photos, feature maps) are embedded as PNG.
import base64, io, re, uuid
import numpy as np
from xml.sax.saxutils import escape, quoteattr
from matplotlib.colors import to_hex, to_rgba
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.text import Annotation
from PIL import Image

S = 5.0                      # px per mm
PT = 0.3528 * S              # px per pt


def col(c):
    r = to_rgba(c)
    return 'none' if r[3] == 0 else to_hex(r, keep_alpha=False)


def opac(c):
    return int(round(to_rgba(c)[3] * 100))


def dash(ls):
    if ls in (None, '-', 'solid'): return ''
    return 'dashed=1;dashPattern=4 2;'


def mathtext(s):
    """Plain-HTML approximation of the mathtext strings used in the figures."""
    if '$' not in s: return escape(s), False
    def conv(m):
        t = m.group(1)
        t = t.replace(r'\Delta', 'Δ').replace(r'\odot', '⊙').replace(r'\oslash', '⊘').replace(r'\sum', 'Σ')
        t = re.sub(r'\\hat\{(\w)\}', lambda q: q.group(1) + '\u0302', t)
        t = t.replace(r'\,', ' ').replace("'", '′').replace(' - ', ' − ')
        t = escape(t)
        t = re.sub(r'_\{([^}]*)\}', r'<sub>\1</sub>', t)
        t = re.sub(r'_(\w)', r'<sub>\1</sub>', t)
        return '<i>' + t + '</i>'
    return re.sub(r'\$([^$]*)\$', conv, escape(s).replace('&amp;', '&')), True


class Doc:
    def __init__(self, name, W, H):
        self.name, self.W, self.H, self.cells, self.n = name, W, H, [], 1

    def _id(self):
        self.n += 1; return f'c{self.n}'

    def vertex(self, z, x, y, w, h, style, value=''):
        self.cells.append((z, f'<mxCell id="{self._id()}" value={quoteattr(value)} style={quoteattr(style)} vertex="1" parent="1">'
                              f'<mxGeometry x="{x * S:.2f}" y="{y * S:.2f}" width="{w * S:.2f}" height="{h * S:.2f}" as="geometry"/></mxCell>'))

    def edge(self, z, pts, style):
        p = [(a * S, b * S) for a, b in pts]
        way = ''.join(f'<mxPoint x="{a:.2f}" y="{b:.2f}"/>' for a, b in p[1:-1])
        self.cells.append((z, f'<mxCell id="{self._id()}" value="" style={quoteattr(style)} edge="1" parent="1">'
                              f'<mxGeometry relative="1" as="geometry"><mxPoint x="{p[0][0]:.2f}" y="{p[0][1]:.2f}" as="sourcePoint"/>'
                              f'<mxPoint x="{p[-1][0]:.2f}" y="{p[-1][1]:.2f}" as="targetPoint"/>'
                              + (f'<Array as="points">{way}</Array>' if way else '') + '</mxGeometry></mxCell>'))

    def xml(self):
        body = ''.join(c for _, c in sorted(self.cells, key=lambda t: t[0]))
        return (f'<mxfile host="app.diagrams.net" type="device"><diagram name={quoteattr(self.name)} id="{uuid.uuid4().hex[:20]}">'
                f'<mxGraphModel dx="{self.W * S:.0f}" dy="{self.H * S:.0f}" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" '
                f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.W * S:.0f}" pageHeight="{self.H * S:.0f}" math="0" shadow="0">'
                f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram></mxfile>')


def png_uri(arr, nearest=False, maxpx=480):
    a = np.asarray(arr)
    if a.dtype != np.uint8: a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a)
    if nearest and max(im.size) < 160:
        k = int(np.ceil(160 / max(im.size))); im = im.resize((im.width * k, im.height * k), Image.NEAREST)
    if max(im.size) > maxpx:
        r = maxpx / max(im.size); im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, 'PNG', optimize=True)
    return 'data:image/png,' + base64.b64encode(b.getvalue()).decode()


def convert(fig, ax, path, W, H, name, font='Arial'):
    d = Doc(name, W, H)
    d.vertex(-1e9, 0, 0, W, H, 'rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=none;locked=1;')
    # images
    for im in ax.images:
        l, r, b, t = im.get_extent()
        x0, x1, y0, y1 = min(l, r), max(l, r), min(b, t), max(b, t)
        arr = im.get_array()
        if hasattr(arr, 'filled'): arr = arr.filled(0)
        if arr.ndim == 2:  # colour-mapped
            arr = im.cmap(im.norm(arr))
        nearest = im.get_interpolation() == 'nearest'
        d.vertex(im.get_zorder(), x0, y0, x1 - x0, y1 - y0,
                 f'shape=image;html=1;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;image={png_uri(arr, nearest)};')
    # patches
    for p in ax.patches:
        z = p.get_zorder(); fc, ec = col(p.get_facecolor()), col(p.get_edgecolor())
        lw = p.get_linewidth() * 0.3528 * S * 0.9 if ec != 'none' else 0
        base = f'html=1;whiteSpace=wrap;fillColor={fc};strokeColor={ec if lw > 0 else "none"};strokeWidth={max(lw, 0.5):.2f};{dash(p.get_linestyle())}'
        if fc != 'none' and opac(p.get_facecolor()) < 100: base += f'fillOpacity={opac(p.get_facecolor())};'
        if isinstance(p, FancyBboxPatch):
            pad = getattr(p.get_boxstyle(), 'pad', 0.0)
            x, y, w, h = p.get_x() - pad, p.get_y() - pad, p.get_width() + 2 * pad, p.get_height() + 2 * pad
            d.vertex(z, x, y, w, h, 'rounded=1;absoluteArcSize=1;arcSize=' + f'{2 * pad * S:.1f};' + base)
        elif isinstance(p, Rectangle):
            x, y, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
            if h < 0: y, h = y + h, -h
            d.vertex(z, x, y, w, h, 'rounded=0;' + base)
        elif isinstance(p, Circle):
            (cx, cy), r = p.center, p.radius
            d.vertex(z, cx - r, cy - r, 2 * r, 2 * r, 'ellipse;' + base)
    # lines
    for ln in ax.lines:
        xs, ys = ln.get_xdata(), ln.get_ydata()
        if len(xs) < 2: continue
        lw = ln.get_linewidth() * 0.3528 * S * 0.9
        d.edge(ln.get_zorder(), list(zip(xs, ys)),
               f'endArrow=none;startArrow=none;html=1;rounded=0;strokeColor={col(ln.get_color())};strokeWidth={max(lw, 0.5):.2f};{dash(ln.get_linestyle())}')
    # arrows and texts
    for t in ax.texts:
        z = t.get_zorder()
        if isinstance(t, Annotation) and t.arrow_patch is not None:
            ap = t.arrowprops; c = col(ap.get('color', '#000000')); lw = ap.get('lw', 1.0) * 0.3528 * S * 0.9
            (x1, y1), (x0, y0) = t.xy, t.xyann
            if np.hypot(x1 - x0, y1 - y0) < 0.3:
                # very short stub: draw only an arrow head
                x0, y0 = x1 - (x1 - x0) / max(np.hypot(x1 - x0, y1 - y0), 1e-6) * 0.6, y1 - (y1 - y0) / max(np.hypot(x1 - x0, y1 - y0), 1e-6) * 0.6
            d.edge(z, [(x0, y0), (x1, y1)], f'endArrow=block;endFill=1;endSize=4;html=1;rounded=0;strokeColor={c};strokeWidth={max(lw, 0.5):.2f};')
        s = t.get_text()
        if not s: continue
        val, ishtml = mathtext(s)
        fs = t.get_fontsize() * PT
        x, y = t.get_position()
        nchar = len(re.sub(r'<[^>]+>', '', val)) or 1
        w = (nchar * fs * 0.56 + 6) / S; h = fs * 1.45 / S
        ha, va = t.get_ha(), t.get_va()
        x0 = x - w / 2 if ha == 'center' else (x if ha == 'left' else x - w)
        y0 = y - h / 2 if va in ('center', 'center_baseline') else (y if va == 'top' else y - h)
        fstyle = (1 if t.get_fontweight() in ('bold', 700, 'heavy') else 0) + (2 if t.get_fontstyle() == 'italic' else 0)
        rot = t.get_rotation()
        style = (f'text;html=1;strokeColor=none;fillColor=none;whiteSpace=nowrap;spacing=0;spacingLeft=2;spacingRight=2;'
                 f'align={"center" if ha == "center" else ha};verticalAlign=middle;fontFamily={font};fontSize={fs:.1f};'
                 f'fontColor={col(t.get_color())};fontStyle={fstyle};' + (f'rotation={-rot:.0f};' if rot else ''))
        d.vertex(z + 0.001, x0, y0, w, h, style, val)
    open(path, 'w', encoding='utf8').write(d.xml())
    return len(d.cells)
