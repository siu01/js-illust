#!/usr/bin/env python3
"""
assets/SD.png を解析して、SVG パスの集合 (shapes.js) を生成する。

パイプライン:
  1. 読み込み → 背景(白)を除去してアルファマスクを作る
  2. k-means で減色し、色クラスタごとのマスクを得る
  3. 連結成分ラベリングでパーツに分割 (面積フィルタで ~TARGET_PARTS 個に絞る)
  4. marching squares (matplotlib) で各パーツの輪郭を抽出 (穴も含む)
  5. Ramer-Douglas-Peucker で頂点を間引き、Catmull-Rom → 3次ベジェで滑らかに
  6. 面積降順 (= 奥から手前) に並べ、位置と色からパーツ名を推定して JSON 出力

出力される shapes.js は座標データのみ。実行時に元画像は一切参照しない。
"""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "assets" / "SD.png"
OUT = ROOT / "shapes.js"

VIEWBOX = 620.0      # 出力座標系
N_COLORS = 36        # 減色数
TARGET_PARTS = 320   # 重要度上位これだけを採用
MIN_AREA = 150       # これ未満の断片は捨てる (元画像 px)
MIN_WEIGHT = 450     # 面積 x 重要度がこれ未満なら捨てる
MIN_HOLE_AREA = 90   # これより小さい穴/島は無視
CLOSE_R = 3          # 色クラスタの交錯をならす closing 半径 (px)
SMOOTH_R = 5         # 減色前のメディアンフィルタ半径 (px)
SMOOTH_MASK = 2.5    # パーツ境界をならすガウシアン sigma (px)
CHROMA_BOOST = 2.2   # Lab の a*b* を強調して色相差を分離しやすくする
MAX_FILL_DIST = 2    # 暗い線のパーツを穴埋めで太らせない最大距離 (px)
MIN_THICKNESS = 2.0  # 最大内接円半径がこれ未満の細い線は捨てる (px)


def disk(r):
    """半径 r の円形構造要素。"""
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def rdp_eps_for(area):
    """小さいパーツほど輪郭を細かく残す。大きいパーツは思い切って間引く。"""
    return min(0.9, max(0.25, math.sqrt(area) * 0.018))


# ---------------------------------------------------------------- 読み込み
def load_rgba():
    img = Image.open(SRC).convert("RGBA")
    a = np.array(img).astype(np.float64)
    rgb, alpha = a[..., :3], a[..., 3]

    # 背景は「画像の外周から白のまま繋がっている領域」だけ。
    # 髪やパーカーの白は被写体の内側にあるので、これなら消えない。
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    near_white = (mn > 250) & ((mx - mn) < 10) & (alpha > 40)

    lab, n = ndimage.label(near_white)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    background = np.isin(lab, list(border))

    fg = (alpha > 40) & (~background)
    fg = ndimage.binary_closing(fg, np.ones((3, 3)))
    fg = ndimage.binary_fill_holes(fg)
    return rgb, fg


def rgb_to_lab(rgb):
    """sRGB (0..255) → CIE Lab。knn を知覚距離で行うために使う。"""
    c = rgb / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = c @ m.T
    white = np.array([0.95047, 1.00000, 1.08883])
    t = xyz / white
    d = 6.0 / 29.0
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


# ------------------------------------------------------------------ 減色
def quantize(rgb, fg):
    # 減色の前に階調をならす。線画のディザやグラデーションのノイズをここで
    # 潰しておかないと、色クラスタが空間的に交錯して破片だらけになる。
    sm = np.dstack([
        ndimage.median_filter(rgb[..., c], size=SMOOTH_R) for c in range(3)
    ])
    sm = np.dstack([
        ndimage.uniform_filter(sm[..., c], size=3) for c in range(3)
    ])

    # RGB の距離で分けると、彩度の高い小さな領域 (瞳の紫) が
    # 髪の薄紫クラスタに吸収されてしまう。Lab に移し、さらに色度 a*b* を
    # 強調して、色相の違いが独立したクラスタになるようにする。
    lab = rgb_to_lab(sm)
    lab[..., 1:] *= CHROMA_BOOST

    px = lab[fg]
    km = KMeans(n_clusters=N_COLORS, n_init=6, random_state=0).fit(px)
    labels = np.full(fg.shape, -1, dtype=np.int32)
    labels[fg] = km.labels_

    # 代表色は平滑化前の実際の画素から取る (平滑化で色が濁るのを避ける)
    palette = np.array([
        rgb[labels == ci].mean(axis=0) if (labels == ci).any() else np.array([128.0, 128.0, 128.0])
        for ci in range(N_COLORS)
    ])
    return labels, palette


def fit_gradient(rgb, mask, scale, offset):
    """
    パーツ内の色を「位置の線形関数」として最小二乗で近似し、
    SVG の linearGradient (両端の座標と色) を推定する。

    面を単色で塗ると階調が段々になるが、これで元のグラデーションを保てる。
    色の変化が小さいパーツは単色として None を返す。
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    cols = rgb[ys, xs]                                  # (n, 3)

    # 各チャンネルを c ≈ a*x + b*y + d で回帰
    A = np.column_stack([xs, ys, np.ones(len(xs))]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, cols, rcond=None)     # (3, 3)

    # 輝度の勾配方向をグラデーションの向きとする
    lum_w = np.array([0.299, 0.587, 0.114])
    gx, gy = coef[0] @ lum_w, coef[1] @ lum_w
    norm = math.hypot(gx, gy)
    if norm < 1e-7:
        return None

    ux, uy = gx / norm, gy / norm
    t = xs * ux + ys * uy
    t0, t1 = np.percentile(t, 2), np.percentile(t, 98)
    if t1 - t0 < 3:
        return None

    # 端の色は「実際にそこにある画素の平均」を使う。回帰式から外挿すると
    # 元画像に存在しない色 (緑やオレンジ) が出てしまう。
    lo = t <= np.percentile(t, 12)
    hi = t >= np.percentile(t, 88)
    if lo.sum() < 8 or hi.sum() < 8:
        return None
    c0 = cols[lo].mean(axis=0)
    c1 = cols[hi].mean(axis=0)
    if np.abs(c1 - c0).max() < 8:                       # ほぼ単色
        return None

    p0 = np.array([t0 * ux, t0 * uy]) * scale + offset
    p1 = np.array([t1 * ux, t1 * uy]) * scale + offset
    hexc = lambda c: "#%02x%02x%02x" % tuple(int(round(v)) for v in c)
    return {
        "x1": round(float(p0[0]), 1), "y1": round(float(p0[1]), 1),
        "x2": round(float(p1[0]), 1), "y2": round(float(p1[1]), 1),
        "c1": hexc(c0), "c2": hexc(c1),
    }


def importance(color):
    """彩度が高い / 暗い色ほど絵の印象を左右するので重みを上げる。"""
    mx, mn = float(max(color)), float(min(color))
    sat = (mx - mn) / 255.0
    dark = 1.0 - mx / 255.0
    return 1.0 + 6.0 * sat + 9.0 * dark * dark


# ------------------------------------------------- 連結成分 → パーツ一覧
def split_parts(labels, palette):
    """
    色クラスタごとに連結成分へ分解し、1 成分 = 1 パーツとして返す。

    closing は穏やかにかけて境界の細かい交錯だけをならす。fill_holes は
    かけない — リング状のクラスタ (顔を囲む髪など) を潰してしまうため。
    パーツ間の隙間は、最背面のシルエットと各パスの同色 stroke で埋める。
    """
    comps = []
    for ci in range(len(palette)):
        mask = labels == ci
        if not mask.any():
            continue
        imp = importance(palette[ci])
        # 減色の境界は細かく入り組んでいて、そのまま輪郭にすると
        # ギザギザの虫食いに見える。ぼかしてから閾値を取って境界をならす。
        # ただし目やまつ毛のような小さく効く部分は強くぼかすと潰れるので、
        # 暗い色・彩度の高い色ほど弱くかける。
        if imp > 3.0:
            mask = ndimage.binary_closing(mask, disk(1))
            mask = ndimage.gaussian_filter(mask.astype(np.float32), 0.8) > 0.5
        else:
            mask = ndimage.binary_closing(mask, disk(CLOSE_R))
            mask = ndimage.gaussian_filter(mask.astype(np.float32), SMOOTH_MASK) > 0.5
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n == 0:
            continue
        areas = ndimage.sum(mask, lab, range(1, n + 1))
        slices = ndimage.find_objects(lab)
        for idx, area in enumerate(areas, start=1):
            if area < MIN_AREA:
                continue
            weight = float(area) * importance(palette[ci])
            if weight < MIN_WEIGHT:
                continue
            # 髪や袖の輪郭線のように細長いマスクは、輪郭を取って平滑化すると
            # 太い塊に化けて原画にない影になる。最大内接円の半径で厚みを測り、
            # 線としか言えないものは落とす。
            sl = slices[idx - 1]
            sub = lab[sl] == idx
            thickness = ndimage.distance_transform_edt(
                np.pad(sub, 1)
            ).max()
            if thickness < MIN_THICKNESS:
                continue
            comps.append({
                "mask_lab": lab,
                "idx": idx,
                "ci": ci,
                "area": float(area),
                "color": palette[ci],
                "weight": weight,
            })

    # 採用するかどうかは「見た目の効き」で決める。目やまつ毛は面積が小さくても
    # 絵の印象を左右するので、彩度・暗さで重み付けして残す。
    comps.sort(key=lambda c: -c["weight"])
    comps = comps[:TARGET_PARTS]
    # 描画順は素の面積降順 (大きい面が奥)
    comps.sort(key=lambda c: -c["area"])
    parts = []
    for depth, c in enumerate(comps):
        parts.append({
            "mask": (c["mask_lab"] == c["idx"]),
            "area": c["area"],
            "color": c["color"],
            "ci": c["ci"],
            "depth": depth,
        })
    return parts


def cover_foreground(parts, fg, labels):
    """
    採用しなかった領域が地色のまま残ると、パーツの隙間から下地が透けて
    汚れに見える。前景を採用パーツだけで完全に分割し直す。

    穴埋めは「同じ色クラスタのパーツ」の中で最近傍を探す。単純な全体最近傍
    にすると、面積の大きい白いパーツが瞳や肌を飲み込んでしまう。
    """
    assign = np.zeros(fg.shape, np.int32)
    # 描画順(奥 → 手前)に塗るので、手前のパーツが重なりを取る
    for i, part in enumerate(parts, start=1):
        assign[part["mask"]] = i

    by_cluster = {}
    for i, part in enumerate(parts, start=1):
        by_cluster.setdefault(part["ci"], []).append(i)

    for ci, ids in by_cluster.items():
        hole = (labels == ci) & fg & (assign == 0)
        if not hole.any():
            continue
        own = np.isin(assign, ids)
        if not own.any():
            continue
        dist, (iy, ix) = ndimage.distance_transform_edt(~own, return_indices=True)
        # 暗い線や瞳のクラスタが遠くまで引っ張ると、細い輪郭線が周囲を
        # 飲み込んで塊になる。そこだけ距離を制限し、広い面は無制限に埋める。
        limit = MAX_FILL_DIST if importance(parts[ids[0] - 1]["color"]) > 3.0 else np.inf
        near = hole & (dist <= limit)
        assign[near] = assign[iy[near], ix[near]]

    # どのパーツも残らなかったクラスタの分だけ、全体の最近傍で埋める
    hole = fg & (assign == 0)
    if hole.any():
        _, (iy, ix) = ndimage.distance_transform_edt(
            assign == 0, return_indices=True
        )
        assign[hole] = assign[iy[hole], ix[hole]]
    assign[~fg] = 0

    kept = []
    for i, part in enumerate(parts, start=1):
        m = assign == i
        a = int(m.sum())
        if a < MIN_AREA:
            continue
        part["mask"] = m
        part["area"] = float(a)
        kept.append(part)
    for depth, part in enumerate(kept):
        part["depth"] = depth
    return kept


# -------------------------------------------------------------- 輪郭抽出
def split_subpaths(path):
    """matplotlib の Path を MOVETO 単位のサブパスに分ける。"""
    v = path.vertices
    codes = path.codes
    if codes is None:
        return [v]
    segs, start = [], 0
    for i, c in enumerate(codes):
        if c == MPath.MOVETO and i > start:
            segs.append(v[start:i])
            start = i
    segs.append(v[start:])
    return [s for s in segs if len(s) >= 3]


def contours_of(mask):
    """marching squares でマスクの輪郭(外周 + 穴)を取り出す。"""
    padded = np.pad(mask.astype(float), 1)
    fig = plt.figure()
    try:
        cs = plt.contour(padded, levels=[0.5])
        paths = []
        for p in cs.get_paths():
            # 1 つの Path が複数のサブパス(外周と穴)を持つ。MOVETO で
            # 分離しないと、別々の輪郭が 1 本の線で繋がってしまう。
            for v in split_subpaths(p):
                if len(v) < 8:
                    continue
                # pad 分を戻す。contour は (x, y) = (col, row) で返る
                pts = np.column_stack([v[:, 0] - 1.0, v[:, 1] - 1.0])
                if polygon_area(pts) < MIN_HOLE_AREA:
                    continue
                paths.append(pts)
        return paths
    finally:
        plt.close(fig)


def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# ------------------------------------------------------ 単純化 + 滑らか化
def rdp(pts, eps):
    """Ramer-Douglas-Peucker。閉曲線なので開いた列として扱う。"""
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    seg = end - start
    rel = pts - start
    cross = seg[0] * rel[:, 1] - seg[1] * rel[:, 0]
    d = np.abs(cross) / (np.linalg.norm(seg) + 1e-9)
    i = int(np.argmax(d))
    if d[i] > eps:
        left = rdp(pts[: i + 1], eps)
        right = rdp(pts[i:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def rdp_closed(pts, eps):
    """
    閉曲線用の RDP。始点と終点が同じ点なので、そのまま rdp にかけると
    基準線の長さが 0 になり全点が潰れてしまう。始点から最も遠い点で
    2 つの開曲線に割ってから簡略化する。戻り値は重複点なしの閉じた点列。
    """
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 4:
        return pts
    far = int(np.argmax(np.linalg.norm(pts - pts[0], axis=1)))
    if far < 2 or far > len(pts) - 2:
        return pts
    a = rdp(pts[: far + 1], eps)
    b = rdp(np.vstack([pts[far:], pts[:1]]), eps)
    return np.vstack([a[:-1], b[:-1]])


def smooth_polygon(pts, iterations=1):
    """閉じた点列に移動平均をかけて、輪郭の細かい波打ちをならす。"""
    if len(pts) < 5:
        return pts
    for _ in range(iterations):
        prev = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
        pts = 0.25 * prev + 0.5 * pts + 0.25 * nxt
    return pts


def to_bezier(pts, scale, offset):
    """閉じた点列を Catmull-Rom 経由で 3 次ベジェの d 文字列にする。"""
    p = pts * scale + offset
    n = len(p)
    if n < 3:
        return ""

    def fmt(v):
        return f"{v:.1f}".rstrip("0").rstrip(".")

    d = [f"M{fmt(p[0][0])} {fmt(p[0][1])}"]
    for i in range(n):
        p0 = p[(i - 1) % n]
        p1 = p[i]
        p2 = p[(i + 1) % n]
        p3 = p[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        d.append(
            f"C{fmt(c1[0])} {fmt(c1[1])} {fmt(c2[0])} {fmt(c2[1])} {fmt(p2[0])} {fmt(p2[1])}"
        )
    d.append("Z")
    return "".join(d)


# ------------------------------------------------------------ パーツ命名
def guess_label(color, cx, cy, area, w, h):
    """色と重心位置からパーツ名を推定する。Live2D 的な部位分けの下地。"""
    r, g, b = color
    mx, mn = max(color), min(color)
    sat = mx - mn
    x = cx / w   # 0..1
    y = cy / h

    if mx < 90:
        return "eyeLine" if y < 0.58 else "legs"
    if sat > 45 and b > r and b > 120 and mx < 210:
        return "iris"
    if r > 210 and g > 175 and b > 150 and sat > 22 and y < 0.60:
        return "skin"
    if r > 200 and g > 165 and b > 135 and sat > 25:
        return "earInner" if y < 0.40 else "skin"
    if y > 0.78:
        return "shoes"
    if 0.55 < y < 0.80 and 0.33 < x < 0.67:
        return "hoodie"
    return "hair"


# ------------------------------------------------------------------ main
def main():
    rgb, fg = load_rgba()
    h, w = fg.shape
    print(f"image {w}x{h}, foreground {fg.sum()} px")

    labels, palette = quantize(rgb, fg)
    print(f"quantized to {len(palette)} colors")

    parts = split_parts(labels, palette)
    print(f"{len(parts)} parts after area filter")
    parts = cover_foreground(parts, fg, labels)
    print(f"{len(parts)} parts after covering the foreground")

    # 元画像 → viewBox へのスケール。被写体を中央に収める
    ys, xs = np.nonzero(fg)
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = bx1 - bx0, by1 - by0
    margin = 0.06
    scale = VIEWBOX * (1 - 2 * margin) / max(bw, bh)
    offset = np.array([
        VIEWBOX / 2 - (bx0 + bw / 2) * scale,
        VIEWBOX / 2 - (by0 + bh / 2) * scale,
    ])

    out = []

    for i, part in enumerate(parts):
        cons = contours_of(part["mask"])
        if not cons:
            continue
        eps = rdp_eps_for(part["area"])
        ds = []
        for pts in cons:
            simp = rdp_closed(pts, eps)
            if len(simp) < 4:
                continue
            ds.append(to_bezier(smooth_polygon(simp), scale, offset))
        if not ds:
            continue

        cy, cx = ndimage.center_of_mass(part["mask"])
        # 色はクラスタ代表色ではなく、このパーツが実際に覆っている画素の
        # 平均から取る。代表色のままだと、同じクラスタに入った離れた領域
        # (髪の毛先の濃い紫と袖口の影など) が同じ色に引きずられる。
        col = rgb[part["mask"]].mean(axis=0)
        entry = {
            "id": f"p{i:03d}",
            "label": guess_label(col, cx, cy, part["area"], w, h),
            "fill": "#%02x%02x%02x" % tuple(int(round(min(255, max(0, c)))) for c in col),
            "depth": part["depth"],
            "area": round(part["area"] * scale * scale, 1),
            "cx": round(cx * scale + offset[0], 1),
            "cy": round(cy * scale + offset[1], 1),
            "d": " ".join(ds),
        }
        grad = fit_gradient(rgb, part["mask"], scale, offset)
        if grad:
            entry["grad"] = grad
        out.append(entry)

    counts = {}
    for s in out:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    print("labels:", counts)

    body = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(
        "// 自動生成: tools/analyze.py が assets/SD.png を解析して出力。\n"
        "// 実行時に画像は参照しない — 以下は座標と色だけのデータ。\n"
        f"window.OKOJO_SHAPES = {body};\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB, {len(out)} shapes)")


if __name__ == "__main__":
    main()
