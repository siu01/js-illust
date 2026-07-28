#!/usr/bin/env python3
"""
assets/SD.png を解析して、SVG パスの集合 (shapes.js) を生成する。

2 パス構成:
  パス1  画像全体を粗く分割する。大きな面 (髪・パーカー・肌) を取る。
  パス2  パス1 の結果を塗り直した「復元画像」と原画の色差を測り、誤差の
         大きい領域だけを ROI として切り出して細かく再解析する。ロゴの
         文字や口のように小さくてコントラストの低い部分は、全体設定では
         平滑化に溶けてしまうため、この 2 パス目で拾う。

各パスの中身:
  1. 背景 (外周から白のまま繋がる領域) を除去して前景マスクを作る
  2. メディアンで階調をならし、Lab 色空間で k-means 減色
  3. 色クラスタを連結成分に分け、面積 x 重要度で採否を決める
  4. marching squares で輪郭を取り、閉曲線 RDP → Catmull-Rom → 3次ベジェ
  5. 未割り当ての画素を同色クラスタの最近傍パーツに吸収させ、前景を完全分割
  6. パーツごとに線形グラデーションを最小二乗で当てる

出力される shapes.js は座標と色だけ。実行時に元画像は一切参照しない。
"""

import json
import math
from dataclasses import dataclass
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
MARGIN = 0.06        # viewBox の余白率
MIN_HOLE_AREA = 90   # これより小さい穴/島は無視


@dataclass
class Pass:
    """1 回の解析パスの設定。ROI ごとに強さを変えるためにまとめている。"""
    n_colors: int          # k-means の色数
    target_parts: int      # 重要度上位これだけを採用
    min_area: int          # これ未満の断片は捨てる (元画像 px)
    min_weight: float      # 面積 x 重要度がこれ未満なら捨てる
    close_r: int           # 色クラスタの交錯をならす closing 半径
    smooth_r: int          # 減色前のメディアンフィルタ半径
    smooth_mask: float     # パーツ境界をならすガウシアン sigma
    min_thickness: float   # 最大内接円半径がこれ未満の細い線は捨てる
    max_fill_dist: float   # 暗い線を穴埋めで太らせない最大距離
    rdp_k: float           # 輪郭単純化の強さ (面積に比例)
    rdp_max: float         # 同 上限


# 画像全体。大きな面をなめらかに取ることを優先する。
BASE = Pass(
    n_colors=36, target_parts=320, min_area=150, min_weight=450,
    close_r=3, smooth_r=5, smooth_mask=2.5, min_thickness=2.0,
    max_fill_dist=2, rdp_k=0.018, rdp_max=0.9,
)

# 残差の大きい ROI。小さく淡い形を拾うため、平滑化と足切りを弱める。
# max_fill_dist=0 は「線のパーツを穴埋めで一切太らせない」の意。ROI には
# まつ毛やフード襟のような細い線が集まっていて、少しでも太らせると原画に
# ない黒い塊になる。空いた分は後段の全体最近傍 (明るい面) が埋める。
REFINE = Pass(
    n_colors=10, target_parts=40, min_area=40, min_weight=90,
    close_r=1, smooth_r=2, smooth_mask=0.8, min_thickness=1.8,
    max_fill_dist=0, rdp_k=0.012, rdp_max=0.5,
)

RESID_THRESH = 9.0    # この Lab 距離を超えたら「再現できていない」とみなす
ROI_MIN_AREA = 260    # これ未満の残差領域は ROI にしない (元画像 px)
ROI_PAD = 10          # ROI の外側に取る余白 (px)
ROI_MAX = 8           # 1 周あたりの ROI 個数上限 (残差の大きい順)
ROI_ROUNDS = 3        # ROI 再解析を繰り返す回数
CHROMA_BOOST = 2.2    # Lab の a*b* を強調して色相差を分離しやすくする

# 線画抽出。LINE_SCALE は「これより細ければ線」とみなす太さの目安。
LINE_SCALE = 7        # black-hat の構造要素半径 (px)
LINE_THRESH = 16      # 周囲との輝度差がこれ以上なら線とみなす
LINE_MIN_AREA = 90    # これ未満の線片は捨てる (元画像 px)
LINE_MAX = 240        # 線パーツの個数上限 (面積の大きい順)
LINE_SMOOTH = 0.8     # 線マスクの角を落とすガウシアン sigma (px)

# 線は細いので、面と同じだけ簡略化・平滑化すると形が崩れる
LINE_PASS = Pass(
    n_colors=1, target_parts=0, min_area=LINE_MIN_AREA, min_weight=0,
    close_r=1, smooth_r=1, smooth_mask=0.0, min_thickness=0.0,
    max_fill_dist=0, rdp_k=0.008, rdp_max=0.35,
)


def disk(r):
    """半径 r の円形構造要素。"""
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


# ---------------------------------------------------------------- 読み込み
def load_rgba():
    img = Image.open(SRC).convert("RGBA")
    a = np.array(img).astype(np.float64)
    rgb, alpha = a[..., :3], a[..., 3]

    # 背景は「画像の外周から白のまま繋がっている領域」だけ。髪やパーカーの
    # 白は被写体の内側にあるので、これなら消えない。閾値を上げているのは、
    # アホ毛のように内部が白い細い形の輪郭線を閉じさせるため。
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    near_white = (mn > 250) & ((mx - mn) < 10) & (alpha > 40)

    lab, _ = ndimage.label(near_white)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    background = np.isin(lab, list(border))

    fg = (alpha > 40) & (~background)
    fg = ndimage.binary_closing(fg, np.ones((3, 3)))
    fg = ndimage.binary_fill_holes(fg)
    return rgb, fg


def rgb_to_lab(rgb):
    """sRGB (0..255) → CIE Lab。色差を知覚的な距離で測るために使う。"""
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
def quantize(rgb, fg, cfg):
    # 減色の前に階調をならす。線画のディザやグラデーションのノイズをここで
    # 潰しておかないと、色クラスタが空間的に交錯して破片だらけになる。
    sm = np.dstack([
        ndimage.median_filter(rgb[..., c], size=cfg.smooth_r) for c in range(3)
    ])
    sm = np.dstack([
        ndimage.uniform_filter(sm[..., c], size=3) for c in range(3)
    ])

    # RGB の距離で分けると、彩度の高い小さな領域 (瞳の紫) が髪の薄紫クラスタ
    # に吸収される。Lab に移し、色度 a*b* を強調して色相差を際立たせる。
    lab = rgb_to_lab(sm)
    lab[..., 1:] *= CHROMA_BOOST

    n_colors = min(cfg.n_colors, max(2, int(fg.sum() // 400)))
    km = KMeans(n_clusters=n_colors, n_init=6, random_state=0).fit(lab[fg])
    labels = np.full(fg.shape, -1, dtype=np.int32)
    labels[fg] = km.labels_

    # 代表色は平滑化前の実際の画素から取る (平滑化で色が濁るのを避ける)
    palette = np.array([
        rgb[labels == ci].mean(axis=0) if (labels == ci).any()
        else np.array([128.0, 128.0, 128.0])
        for ci in range(n_colors)
    ])
    return labels, palette


def importance(color):
    """彩度が高い / 暗い色ほど絵の印象を左右するので重みを上げる。"""
    mx, mn = float(max(color)), float(min(color))
    sat = (mx - mn) / 255.0
    dark = 1.0 - mx / 255.0
    return 1.0 + 6.0 * sat + 9.0 * dark * dark


# ------------------------------------------------- 連結成分 → パーツ一覧
def split_parts(labels, palette, cfg):
    """色クラスタごとに連結成分へ分解し、1 成分 = 1 パーツとして返す。"""
    comps = []
    for ci in range(len(palette)):
        mask = labels == ci
        if not mask.any():
            continue
        # 減色の境界は細かく入り組んでいて、そのまま輪郭にするとギザギザの
        # 虫食いに見える。ぼかしてから閾値を取って境界をならす。ただし目や
        # まつ毛は強くぼかすと潰れるので、暗い / 彩度の高い色ほど弱くかける。
        if importance(palette[ci]) > 3.0:
            mask = ndimage.binary_closing(mask, disk(1))
            mask = ndimage.gaussian_filter(mask.astype(np.float32), 0.8) > 0.5
        else:
            mask = ndimage.binary_closing(mask, disk(cfg.close_r))
            sig = cfg.smooth_mask
            mask = ndimage.gaussian_filter(mask.astype(np.float32), sig) > 0.5

        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n == 0:
            continue
        areas = ndimage.sum(mask, lab, range(1, n + 1))
        slices = ndimage.find_objects(lab)
        for idx, area in enumerate(areas, start=1):
            if area < cfg.min_area:
                continue
            weight = float(area) * importance(palette[ci])
            if weight < cfg.min_weight:
                continue
            # 輪郭線のように細長いマスクは、輪郭を取って平滑化すると太い塊に
            # 化けて原画にない影になる。最大内接円の半径で厚みを測って落とす。
            sub = lab[slices[idx - 1]] == idx
            if ndimage.distance_transform_edt(np.pad(sub, 1)).max() < cfg.min_thickness:
                continue
            comps.append({
                "mask_lab": lab, "idx": idx, "ci": ci,
                "area": float(area), "color": palette[ci], "weight": weight,
            })

    # 採否は「見た目の効き」で決める。目やまつ毛は面積が小さくても絵の印象を
    # 左右するので、彩度・暗さで重み付けして残す。
    comps.sort(key=lambda c: -c["weight"])
    comps = comps[:cfg.target_parts]
    comps.sort(key=lambda c: -c["area"])       # 描画順は面積降順 (大きい面が奥)

    return [{
        "mask": (c["mask_lab"] == c["idx"]),
        "area": c["area"], "color": c["color"], "ci": c["ci"], "depth": depth,
    } for depth, c in enumerate(comps)]


def cover_foreground(parts, fg, labels, cfg):
    """
    採用しなかった領域が地色のまま残ると、パーツの隙間から下地が透けて汚れに
    見える。前景を採用パーツだけで完全に分割し直す。

    穴埋めは「同じ色クラスタのパーツ」の中で最近傍を探す。単純な全体最近傍に
    すると、面積の大きい白いパーツが瞳や肌を飲み込んでしまう。
    """
    assign = np.zeros(fg.shape, np.int32)
    for i, part in enumerate(parts, start=1):    # 奥 → 手前。手前が重なりを取る
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
        # 暗い線や瞳のクラスタが遠くまで引っ張ると、細い輪郭線が周囲を飲み
        # 込んで塊になる。そこだけ距離を制限し、広い面は無制限に埋める。
        limit = (cfg.max_fill_dist
                 if importance(parts[ids[0] - 1]["color"]) > 3.0 else np.inf)
        near = hole & (dist <= limit)
        assign[near] = assign[iy[near], ix[near]]

    # どのパーツも残らなかったクラスタの分だけ、全体の最近傍で埋める
    hole = fg & (assign == 0)
    if hole.any():
        _, (iy, ix) = ndimage.distance_transform_edt(assign == 0, return_indices=True)
        assign[hole] = assign[iy[hole], ix[hole]]
    assign[~fg] = 0

    kept = []
    for i, part in enumerate(parts, start=1):
        m = assign == i
        a = int(m.sum())
        if a < cfg.min_area:
            continue
        part["mask"] = m
        part["area"] = float(a)
        kept.append(part)
    for depth, part in enumerate(kept):
        part["depth"] = depth
    return kept


def analyze_region(rgb, fg, cfg, box=None):
    """fg (必要なら box で切った範囲) を解析してパーツ一覧を返す。"""
    region = fg
    if box is not None:
        y0, y1, x0, x1 = box
        region = np.zeros_like(fg)
        region[y0:y1, x0:x1] = fg[y0:y1, x0:x1]
    if region.sum() < cfg.min_area * 2:
        return []

    labels, palette = quantize(rgb, region, cfg)
    parts = split_parts(labels, palette, cfg)
    if not parts:
        return []
    parts = cover_foreground(parts, region, labels, cfg)
    for part in parts:                       # 実測平均色を確定させる
        part["color"] = rgb[part["mask"]].mean(axis=0)
    return parts


# ------------------------------------------------------------ 線画の抽出
def extract_lines(rgb, fg):
    """
    原画は「線画 + 塗り」で描かれている。面だけを再現しても線が落ちるので、
    ここで線だけを取り出す。

    black-hat (グレースケールの closing との差) は「周囲より暗い細い構造」に
    反応する。線幅より少し大きい構造要素を使うので、広い影には反応せず、
    髪の毛の流れやまつ毛のような線だけが残る。
    """
    gray = rgb.mean(axis=2)
    black_hat = ndimage.grey_closing(gray, footprint=disk(LINE_SCALE)) - gray
    lines = fg & (black_hat > LINE_THRESH)
    # 1px のごま塩を落とす。線そのものは細いので opening は最小限に
    lines = ndimage.binary_opening(lines, disk(1))
    # 細いマスクの輪郭はそのままだと往復して棘だらけになる。閾値 0.5 の
    # ガウシアンで角を落とす (太さは変えない)。
    lines = ndimage.gaussian_filter(lines.astype(np.float32), LINE_SMOOTH) > 0.5
    return lines


def line_parts(rgb, lines, depth_base):
    """線画マスクを連結成分に分け、そのままの太さでパーツにする。"""
    lab, n = ndimage.label(lines, structure=np.ones((3, 3)))
    if n == 0:
        return []
    areas = ndimage.sum(lines, lab, range(1, n + 1))
    parts = []
    for idx, area in enumerate(areas, start=1):
        if area < LINE_MIN_AREA:
            continue
        mask = lab == idx
        parts.append({
            "mask": mask,
            "area": float(area),
            "color": rgb[mask].mean(axis=0),
            "ci": -1,
            "depth": 0,
        })
    # 線は太いものから順に。細い線が上に来るようにする
    parts.sort(key=lambda p: -p["area"])
    for i, part in enumerate(parts[:LINE_MAX]):
        part["depth"] = depth_base + i
    return parts[:LINE_MAX]


def residual_map(rgb, fg, parts):
    """パーツを塗り直した復元画像と原画の、画素ごとの Lab 距離。"""
    recon = np.zeros_like(rgb)
    for part in parts:            # parts は奥 → 手前の順。手前が上書きする
        recon[part["mask"]] = part["color"]
    return np.linalg.norm(rgb_to_lab(rgb) - rgb_to_lab(recon), axis=-1)


def report_residual(rgb, fg, parts, tag):
    """
    再現できていない度合いを数値で出す。改善が測れないと止め時が分からない。

    ただしこれは面積で平均した指標なので、面の再現度を見るためのもの。線画を
    重ねた後はむしろ悪化することがある — 線は面積が小さく、1px ずれるだけで
    両側に誤差が出るため。見た目の良し悪しはこの数値では決まらない。
    """
    d = residual_map(rgb, fg, parts)[fg]
    print(f"    residual {tag}: mean dE {d.mean():.2f}  "
          f"p95 {np.percentile(d, 95):.2f}  "
          f"over-threshold {100 * (d > RESID_THRESH).mean():.1f}%")


def residual_rois(rgb, fg, parts):
    """
    パーツを塗り直した復元画像と原画の色差を測り、誤差の大きいかたまりを
    ROI (再解析する矩形) として返す。どこが再現できていないかを画像自身に
    決めさせるので、顔やロゴの位置を決め打ちしなくて済む。
    """
    diff = residual_map(rgb, fg, parts)
    bad = fg & (diff > RESID_THRESH)
    bad = ndimage.binary_opening(bad, disk(2))
    bad = ndimage.binary_closing(bad, disk(5))

    lab, n = ndimage.label(bad, structure=np.ones((3, 3)))
    if n == 0:
        return []
    areas = ndimage.sum(bad, lab, range(1, n + 1))
    sums = ndimage.sum(diff * bad, lab, range(1, n + 1))
    slices = ndimage.find_objects(lab)

    cands = []
    h, w = fg.shape
    for idx in range(1, n + 1):
        if areas[idx - 1] < ROI_MIN_AREA:
            continue
        sy, sx = slices[idx - 1]
        box = (
            max(0, sy.start - ROI_PAD), min(h, sy.stop + ROI_PAD),
            max(0, sx.start - ROI_PAD), min(w, sx.stop + ROI_PAD),
        )
        cands.append((sums[idx - 1], box))

    cands.sort(key=lambda c: -c[0])
    return [box for _, box in cands[:ROI_MAX]]


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
            # 1 つの Path が複数のサブパス(外周と穴)を持つ。MOVETO で分離
            # しないと、別々の輪郭が 1 本の線で繋がってしまう。
            for v in split_subpaths(p):
                if len(v) < 8:
                    continue
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
    """Ramer-Douglas-Peucker。開いた点列を前提とする。"""
    if len(pts) < 3:
        return pts
    start, end = pts[0], pts[-1]
    seg = end - start
    rel = pts - start
    cross = seg[0] * rel[:, 1] - seg[1] * rel[:, 0]
    d = np.abs(cross) / (np.linalg.norm(seg) + 1e-9)
    i = int(np.argmax(d))
    if d[i] > eps:
        return np.vstack([rdp(pts[: i + 1], eps)[:-1], rdp(pts[i:], eps)])
    return np.vstack([start, end])


def rdp_closed(pts, eps):
    """
    閉曲線用の RDP。始点と終点が同じ点なので、そのまま rdp にかけると基準線
    の長さが 0 になり全点が潰れる。始点から最も遠い点で 2 つの開曲線に割って
    から簡略化する。戻り値は重複点なしの閉じた点列。
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
        p0, p1 = p[(i - 1) % n], p[i]
        p2, p3 = p[(i + 1) % n], p[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        d.append(f"C{fmt(c1[0])} {fmt(c1[1])} {fmt(c2[0])} {fmt(c2[1])} "
                 f"{fmt(p2[0])} {fmt(p2[1])}")
    d.append("Z")
    return "".join(d)


def fit_gradient(rgb, mask, scale, offset):
    """
    パーツ内の色を位置の線形関数として最小二乗で近似し、SVG の
    linearGradient (両端の座標と色) を推定する。単色で塗ると階調が段々に
    なるが、これで元のグラデーションを保てる。変化が小さければ None。
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    cols = rgb[ys, xs]

    A = np.column_stack([xs, ys, np.ones(len(xs))]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, cols, rcond=None)

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
    lo, hi = t <= np.percentile(t, 12), t >= np.percentile(t, 88)
    if lo.sum() < 8 or hi.sum() < 8:
        return None
    c0, c1 = cols[lo].mean(axis=0), cols[hi].mean(axis=0)
    if np.abs(c1 - c0).max() < 8:
        return None

    p0 = np.array([t0 * ux, t0 * uy]) * scale + offset
    p1 = np.array([t1 * ux, t1 * uy]) * scale + offset
    hexc = lambda c: "#%02x%02x%02x" % tuple(int(round(v)) for v in c)
    return {
        "x1": round(float(p0[0]), 1), "y1": round(float(p0[1]), 1),
        "x2": round(float(p1[0]), 1), "y2": round(float(p1[1]), 1),
        "c1": hexc(c0), "c2": hexc(c1),
    }


# ------------------------------------------------------------ パーツ命名
def guess_label(color, cx, cy, w, h):
    """色と重心位置からパーツ名を推定する。Live2D 的な部位分けの下地。"""
    r, g, b = color
    mx, mn = max(color), min(color)
    sat = mx - mn
    x, y = cx / w, cy / h

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


def to_shape(part, index, rgb, scale, offset, w, h, cfg):
    """パーツ 1 つを shapes.js のエントリに変換する。"""
    cons = contours_of(part["mask"])
    if not cons:
        return None
    eps = min(cfg.rdp_max, max(0.25, math.sqrt(part["area"]) * cfg.rdp_k))
    ds = []
    for pts in cons:
        simp = rdp_closed(pts, eps)
        if len(simp) < 4:
            continue
        ds.append(to_bezier(smooth_polygon(simp), scale, offset))
    if not ds:
        return None

    cy, cx = ndimage.center_of_mass(part["mask"])
    col = part["color"]
    entry = {
        "id": f"p{index:03d}",
        "label": guess_label(col, cx, cy, w, h),
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
    return entry


# ------------------------------------------------------------------ main
def main():
    rgb, fg = load_rgba()
    h, w = fg.shape
    print(f"image {w}x{h}, foreground {int(fg.sum())} px")

    # --- パス1: 全体 ---
    parts = analyze_region(rgb, fg, BASE)
    print(f"pass 1: {len(parts)} parts")
    for part in parts:
        part["cfg"] = BASE
    report_residual(rgb, fg, parts, "after pass 1")

    # --- パス2 以降: 残差の大きい ROI だけ細かく、を繰り返す ---
    # 1 周ごとに直せた場所の残差は下がるので、次に効く場所が自動的に浮上する。
    for round_i in range(1, ROI_ROUNDS + 1):
        rois = residual_rois(rgb, fg, parts)
        if not rois:
            print(f"round {round_i}: no residual ROI left")
            break
        added = 0
        for box in rois:
            sub = analyze_region(rgb, fg, REFINE, box)
            if not sub:
                continue
            # ROI 内は完全分割されているので、そのまま最前面に重ねれば
            # 前のパスのパーツを覆い隠す
            base_depth = len(parts)
            for part in sub:
                part["cfg"] = REFINE
                part["depth"] += base_depth
            parts.extend(sub)
            added += len(sub)
        print(f"round {round_i}: {len(rois)} ROIs -> +{added} parts "
              f"(total {len(parts)})")
        report_residual(rgb, fg, parts, f"after round {round_i}")

    # --- 最後に線画を最前面へ ---
    # 残差マップを見ると、面はほぼ再現できていて、残っているのはほとんどが
    # 線画だった。線を面として太らせずに、そのままの太さで重ねる。
    lines = extract_lines(rgb, fg)
    lparts = line_parts(rgb, lines, len(parts))
    for part in lparts:
        part["cfg"] = LINE_PASS
    parts.extend(lparts)
    print(f"lines: +{len(lparts)} parts (total {len(parts)})")
    report_residual(rgb, fg, parts, "after lines")

    # 元画像 → viewBox へのスケール。被写体を中央に収める
    ys, xs = np.nonzero(fg)
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = bx1 - bx0, by1 - by0
    scale = VIEWBOX * (1 - 2 * MARGIN) / max(bw, bh)
    offset = np.array([
        VIEWBOX / 2 - (bx0 + bw / 2) * scale,
        VIEWBOX / 2 - (by0 + bh / 2) * scale,
    ])

    out = []
    for i, part in enumerate(parts):
        entry = to_shape(part, i, rgb, scale, offset, w, h, part["cfg"])
        if entry:
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
