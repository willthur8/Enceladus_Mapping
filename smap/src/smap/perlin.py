
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from skimage import measure


QUAD = 64

def perlin_grid(res, rng):
    ry, rx = res
    H, W = perlin_grid.shape

    # corner gradients
    ang = 2 * np.pi * rng.random((ry + 1, rx + 1))
    grad = np.dstack((np.cos(ang), np.sin(ang)))

    # sample coords
    ys = np.linspace(0, ry, H, endpoint=False)
    xs = np.linspace(0, rx, W, endpoint=False)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")

    y0 = gy.astype(int); x0 = gx.astype(int)
    fy = gy - y0;        fx = gx - x0

    def dot(iy, ix, oy, ox):
        g = grad[(y0 + iy), (x0 + ix)]
        return g[..., 0] * (fy - oy) + g[..., 1] * (fx - ox)

    n00 = dot(0, 0, 0, 0)
    n01 = dot(0, 1, 0, 1)
    n10 = dot(1, 0, 1, 0)
    n11 = dot(1, 1, 1, 1)

    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    u = fade(fx); v = fade(fy)
    top = n00 * (1 - u) + n01 * u
    bot = n10 * (1 - u) + n11 * u
    return top * (1 - v) + bot * v


def fractal_noise(shape, rng, octaves=5, base_res=3, persistence=0.5, lacunarity=2):

    perlin_grid.shape = shape
    field = np.zeros(shape)
    amp = 1.0
    freq = base_res
    norm = 0.0
    for _ in range(octaves):
        field += amp * perlin_grid((freq, freq), rng)
        norm += amp
        amp *= persistence
        freq = int(freq * lacunarity)
    return field / norm

def mask_to_polygons(field, level, extent):

    H, W = field.shape
    minx, maxx, miny, maxy = extent
    mask = np.pad((field > level).astype(float), 1, constant_values=0.0)

    loops = []
    for c in measure.find_contours(mask, 0.5):

        rr = c[:, 0] - 1
        cc = c[:, 1] - 1
        x = minx + (cc / (W - 1)) * (maxx - minx)
        y = miny + (rr / (H - 1)) * (maxy - miny)
        ring = Polygon(np.column_stack([x, y]))
        if ring.is_valid and ring.area > 0:
            loops.append(ring)

    if not loops:
        return Polygon()
    reps = [lp.representative_point() for lp in loops]
    depth = [sum(other.contains(reps[i]) for k, other in enumerate(loops) if k != i)
             for i in range(len(loops))]
    shells = [i for i in range(len(loops)) if depth[i] % 2 == 0]
    holes = [i for i in range(len(loops)) if depth[i] % 2 == 1]

    polys = []
    for i in shells:
        my_holes = []
        for h in holes:
            if not loops[i].contains(reps[h]):
                continue
            parent = min(
                (s for s in shells if loops[s].contains(reps[h])),
                key=lambda s: loops[s].area,
            )
            if parent == i:
                my_holes.append(np.array(loops[h].exterior.coords))
        polys.append(Polygon(np.array(loops[i].exterior.coords), my_holes))

    return unary_union([p.buffer(0) for p in polys])


def make_perlin_splat(domain,rng,grid=400,
    octaves=5, #lower smoother
    base_res=3,
    persistence=0.5, #lower smoother
    threshold=0.7, # size - higher shirnks
    falloff=1.6,
    min_area_frac=0.12,
):

    R = np.sqrt(domain.area / np.pi)
    field = fractal_noise((grid, grid), rng, octaves=octaves,
                          base_res=base_res, persistence=persistence)

    # normalize to [0, 1]
    field = (field - field.min()) / (np.ptp(field) + 1e-12)

    # subtract a bowl that rises toward the edge
    if falloff > 0:
        lin = np.linspace(-1, 1, grid)
        xx, yy = np.meshgrid(lin, lin)
        rr = np.sqrt(xx**2 + yy**2)
        field = field - (rr ** falloff)
        field = (field - field.min()) / (np.ptp(field) + 1e-12)

    extent = (-R, R, -R, R)
    splat = mask_to_polygons(field, threshold, extent)
    splat = splat.intersection(domain).buffer(0)

    # keep only good blobs
    if isinstance(splat, MultiPolygon):
        keep = [g for g in splat.geoms if g.area >= min_area_frac * domain.area]
        splat = unary_union(keep) if keep else max(splat.geoms, key=lambda g: g.area)

    return splat
