"""
線画マスクを「面」ではなく「ストローク」に変換する。

面として塗ると、細いマスクの輪郭が往復して棘になり、それを平滑化で消そうと
すると細い線ごと消えてしまう ("棘を消す" と "細い線を残す" が同じつまみになる)。
中心線と線幅に分けて持てば、この二つが独立する。変形させたときも線幅が
追従するので、部位ごとのアニメーションと相性がよい。

  1. Zhang-Suen で 1px の中心線まで細める
  2. 端点と分岐点で切って、枝ごとのポリラインにする
  3. 各ポリラインの線幅は、距離変換の値 (最大内接円の半径) の 2 倍
"""

import math

import numpy as np
from scipy import ndimage


# 8 近傍を (P2..P9) = 北, 北東, 東, 南東, 南, 南西, 西, 北西 の順で並べる。
# Zhang-Suen の A(P1) はこの巡回順での 0→1 遷移数として定義されている。
_NEIGHBOR_OFFSETS = [
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1),
]

# 分岐点で線を繋ぐかどうかの閾値。進行方向との cos がこれ未満なら、
# 曲がりすぎているので別の線とみなして切る。
STRAIGHT_MIN = 0.35


def _neighbors(img):
    """P2..P9 を平面ごとに積んだ (8, H, W) 配列を返す。"""
    return np.stack([
        np.roll(np.roll(img, -dy, axis=0), -dx, axis=1)
        for dy, dx in _NEIGHBOR_OFFSETS
    ])


def thin(mask):
    """Zhang-Suen 細線化。1px 幅の中心線を返す。"""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        removed = False
        for step in (0, 1):
            p = _neighbors(img)
            # B(P1): 8 近傍のうち前景の数
            b = p.sum(axis=0)
            # A(P1): P2→P3→...→P9→P2 と回ったときの 0→1 遷移数
            nxt = np.roll(p, -1, axis=0)
            a = ((p == 0) & (nxt == 1)).sum(axis=0)

            p2, p4, p6, p8 = p[0], p[2], p[4], p[6]
            if step == 0:
                c1 = p2 * p4 * p6
                c2 = p4 * p6 * p8
            else:
                c1 = p2 * p4 * p8
                c2 = p2 * p6 * p8

            drop = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & (c1 == 0) & (c2 == 0)
            if drop.any():
                img[drop] = 0
                removed = True
        if not removed:
            break
    return img[1:-1, 1:-1].astype(bool)


def _connectivity(skel):
    """細線化した画像の各画素について、8 近傍にある前景画素の数。"""
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    return ndimage.convolve(skel.astype(np.int32), k, mode="constant")


def trace_polylines(skel, min_length=6):
    """
    中心線を、端点と分岐点で切った枝ごとのポリラインに分解する。

    分岐点をまたいで繋げてしまうと、髪が交差するところで線が飛ぶ。切って
    おけば、あとで部位ごとに割り当てるときも枝単位で扱える。
    """
    skel = skel.copy()
    deg = _connectivity(skel) * skel
    # 次数 1 が端点、3 以上が分岐点。分岐点は枝の切れ目として扱う
    nodes = skel & ((deg == 1) | (deg >= 3))

    visited = np.zeros_like(skel)
    polylines = []

    def neighbors_of(y, x):
        out = []
        for dy, dx in _NEIGHBOR_OFFSETS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and skel[ny, nx]:
                out.append((ny, nx))
        return out

    def walk(start, second):
        """
        start から second の向きに、行き止まりまで辿る。

        分岐点で機械的に切ると、髪のように線が何度も交差する絵では 1 本の
        流れが短い断片に割れてしまう。分岐では「それまでの進行方向に最も近い
        枝」を選んで進み、線を 1 本に保つ。
        """
        path = [start, second]
        seen = {start, second}
        if deg[second] < 3:
            visited[second] = True
        prev, cur = start, second
        while True:
            if deg[cur] == 1 and len(path) > 2:      # 端点に着いた
                break
            # 分岐点は複数の線が通る場所なので visited で占有しない。占有すると
            # 先に通った線が交差点を取ってしまい、他の線がそこで途切れる。
            # 同じ線が同じ画素に戻るのは seen で防ぐ。
            cands = [n for n in neighbors_of(*cur)
                     if n != prev and n not in seen and not visited[n]]
            if not cands:
                break

            vy, vx = cur[0] - prev[0], cur[1] - prev[1]
            vn = math.hypot(vy, vx) or 1.0

            def cosine(n):
                wy, wx = n[0] - cur[0], n[1] - cur[1]
                wn = math.hypot(wy, wx) or 1.0
                return (vy * wy + vx * wx) / (vn * wn)

            if deg[cur] >= 3:
                # 分岐点。まっすぐ続く枝があればそれを辿り、無ければここで
                # 終える。どれかに無理やり曲がると、別の線に飛び移ってしまう。
                best = max(cands, key=cosine)
                if cosine(best) < STRAIGHT_MIN:
                    break
                cands = [best]
            elif len(cands) > 1:
                cands.sort(key=cosine, reverse=True)

            prev, cur = cur, cands[0]
            if deg[cur] < 3:
                visited[cur] = True
            seen.add(cur)
            path.append(cur)
        return path

    # 端点から辿る。端から入ったほうが線が途中で切れにくい
    for y, x in list(zip(*np.nonzero(deg == 1))) + list(zip(*np.nonzero(deg >= 3))):
        if not skel[y, x]:
            continue
        for n in neighbors_of(y, x):
            if visited[n]:
                continue
            visited[(y, x)] = True
            path = walk((y, x), n)
            if len(path) >= min_length:
                polylines.append(np.array(path, dtype=np.float64))

    # 節点を持たない閉ループが残るので、それも拾う
    leftover = skel & ~visited & ~nodes
    lab, n = ndimage.label(leftover, structure=np.ones((3, 3)))
    for idx in range(1, n + 1):
        ys, xs = np.nonzero(lab == idx)
        if len(ys) < min_length:
            continue
        start = (int(ys[0]), int(xs[0]))
        visited[start] = True
        cur, path = start, [start]
        while True:
            cands = [p for p in neighbors_of(*cur) if not visited[p] and lab[p] == idx]
            if not cands:
                break
            cur = cands[0]
            visited[cur] = True
            path.append(cur)
        if len(path) >= min_length:
            polylines.append(np.array(path, dtype=np.float64))

    return polylines


def strokes_from_mask(mask, rgb, min_length=6, min_width=0.8):
    """
    線画マスクを、中心線 (row, col の点列) と線幅と色の組に変換する。

    線幅は距離変換 — 中心線上の値がそのまま「線の中心から縁までの距離」に
    なるので、2 倍すれば線幅になる。
    """
    dist = ndimage.distance_transform_edt(mask)
    skel = thin(mask)
    out = []
    for pts in trace_polylines(skel, min_length):
        idx = (pts[:, 0].astype(int), pts[:, 1].astype(int))
        width = float(np.median(dist[idx])) * 2.0
        if width < min_width:
            continue
        cols = rgb[idx]
        out.append({
            "points": pts,                      # (n, 2) の (row, col)
            "width": width,
            "color": cols.mean(axis=0),
            "length": float(len(pts)),
        })
    return out
