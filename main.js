// main.js
// shapes.js (tools/analyze.py が assets/SD.png を解析して生成した座標データ) から
// SVG を組み立てて描画する。実行時に画像は一切読み込まない。
//
// パーツは推定ラベル(hair / skin / iris / hoodie / legs ...)ごとに <g> へまとめ、
// Live2D 的に部位単位でアニメーションできるようにしている。

const NS = "http://www.w3.org/2000/svg";

// まつ毛と瞳が作る外形の、この割合ぶんを「目の内側」とみなす。
// 大きくすると頬まで巻き込み、小さくすると白目が取り残される。
const EYE_INNER_RATIO = 0.42;

// 口の縦スケールの下限と上限
const MOUTH_CLOSED = 0.55;
const MOUTH_OPEN = 2.6;

function el(tag, attrs) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

function build() {
  const shapes = window.OKOJO_SHAPES || [];
  const svg = el("svg", {
    id: "okojo",
    viewBox: "0 0 620 620",
    role: "img",
    "aria-label": "OKOJO SD character drawn in JavaScript",
  });

  // グラデーションを持つパーツ用の <defs>
  const defs = el("defs", {});
  svg.appendChild(defs);

  // 描画順は解析時に決めた depth (奥 → 手前)。
  // ラベルは data-label に残しておき、アニメーション時の部位選択に使う。
  for (const s of [...shapes].sort((a, b) => a.depth - b.depth || b.area - a.area)) {
    // 線画は中心線 + 線幅のストローク。面のように塗りつぶさない。
    if (s.stroke) {
      const attrs = {
        id: s.id,
        "data-label": s.label,
        d: s.d,
        fill: "none",
        stroke: s.stroke,
        "stroke-width": s.width,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
      };
      if (s.opacity !== undefined) attrs.opacity = s.opacity;
      svg.appendChild(el("path", attrs));
      continue;
    }

    let paint = s.fill;

    if (s.grad) {
      const gid = "g-" + s.id;
      const lg = el("linearGradient", {
        id: gid,
        gradientUnits: "userSpaceOnUse",
        x1: s.grad.x1, y1: s.grad.y1,
        x2: s.grad.x2, y2: s.grad.y2,
      });
      lg.appendChild(el("stop", { offset: "0", "stop-color": s.grad.c1 }));
      lg.appendChild(el("stop", { offset: "1", "stop-color": s.grad.c2 }));
      defs.appendChild(lg);
      paint = `url(#${gid})`;
    }

    const attrs = {
      id: s.id,
      "data-label": s.label,
      d: s.d,
      fill: paint,
      "fill-rule": "evenodd",
      // 隣接パーツの継ぎ目に地の色が出ないよう、同じ塗りで細く縁取る
      stroke: paint,
      "stroke-width": "0.8",
      "stroke-linejoin": "round",
    };
    if (s.opacity !== undefined) attrs.opacity = s.opacity;

    svg.appendChild(el("path", attrs));
  }

  return svg;
}

// --------------------------------------------------------------- 部位分け
// 同じラベルのパーツをまとめて <g> に入れ直す。<g> ごと transform すれば
// 部位単位で動かせる。回転や拡縮の原点はその部位の重心に置く。
function groupParts(svg, label) {
  const nodes = [...svg.querySelectorAll(`[data-label="${label}"]`)];
  if (!nodes.length) return null;

  const shapes = (window.OKOJO_SHAPES || []).filter((s) => s.label === label);
  const cx = shapes.reduce((a, s) => a + s.cx, 0) / shapes.length;
  const cy = shapes.reduce((a, s) => a + s.cy, 0) / shapes.length;

  // 元の重なり順を壊さないよう、最前面のパーツがいた場所にグループを差し込む
  const g = el("g", { "data-part": label });
  nodes[nodes.length - 1].after(g);
  for (const n of nodes) g.appendChild(n);

  g.dataset.cx = cx;
  g.dataset.cy = cy;
  return g;
}

// 左右の目は別々に動かしたいので、中心より左か右かで分ける
function groupEyes(svg) {
  const all = window.OKOJO_SHAPES || [];
  const nodes = [...svg.querySelectorAll('[data-label="eye"]')];
  if (!nodes.length) return [];
  const shapes = all.filter((s) => s.label === "eye");
  const mid = shapes.reduce((a, s) => a + s.cx, 0) / shapes.length;

  const sides = [[], []];
  for (const n of nodes) {
    const s = shapes.find((x) => x.id === n.id);
    if (s) sides[s.cx < mid ? 0 : 1].push([n, s]);
  }

  const byId = new Map(all.map((s) => [s.id, s]));

  return sides.filter((side) => side.length).map((side, i) => {
    const g = el("g", { "data-part": "eye" + (i ? "R" : "L") });
    side[side.length - 1][0].after(g);
    for (const [n] of side) g.appendChild(n);

    // ここまでに入っているのは暗いパーツ (まつ毛と瞳) だけ。白目は肌より
    // 明るいので色では拾えず、置き去りにすると目を閉じたときに白い穴が残る。
    // まつ毛と瞳が作る外形の内側にあるものを、明るさに関係なく取り込む。
    // 頬はこの外形の下にあるので入らない。
    const b = g.getBBox();
    const cx = b.x + b.width / 2;
    const cy = b.y + b.height / 2;
    const rx = b.width * EYE_INNER_RATIO;
    const ry = b.height * EYE_INNER_RATIO;
    for (const s of all) {
      if (s.label === "eye" || s.label === "mouth" || s.label === "ahoge") continue;
      if (Math.abs(s.cx - cx) < rx && Math.abs(s.cy - cy) < ry) {
        const n = svg.getElementById(s.id);
        if (n && n.parentNode !== g) g.appendChild(n);
      }
    }

    // 取り込みで重なり順が崩れるので、解析時の depth で並べ直す
    const ordered = [...g.children].sort(
      (a, z) => (byId.get(a.id)?.depth ?? 0) - (byId.get(z.id)?.depth ?? 0)
    );
    for (const n of ordered) g.appendChild(n);

    g.dataset.cx = cx;
    g.dataset.cy = cy;
    return g;
  });
}

// ------------------------------------------------------------ アニメーション
function animate(svg) {
  const eyes = groupEyes(svg);
  const mouth = groupParts(svg, "mouth");
  const ahoge = groupParts(svg, "ahoge");
  if (!eyes.length && !mouth && !ahoge) return;

  const at = (g) => [parseFloat(g.dataset.cx), parseFloat(g.dataset.cy)];

  // まばたきは「たまに、素早く」。等間隔だと機械的に見える
  let nextBlink = 1.2;
  let blinkStart = -1;
  const BLINK = 0.13;                       // 閉じて開くまで

  const t0 = performance.now();
  function frame(now) {
    const t = (now - t0) / 1000;

    if (t > nextBlink && blinkStart < 0) blinkStart = t;
    let lid = 1;
    if (blinkStart >= 0) {
      const p = (t - blinkStart) / BLINK;
      if (p >= 1) {
        blinkStart = -1;
        // 次は 2〜6 秒後。ときどき二回続けて瞬きする
        nextBlink = t + (Math.random() < 0.22 ? 0.22 : 2 + Math.random() * 4);
      } else {
        // 閉じ切りで 0 になる三角波。開くほうを少し緩やかに
        lid = p < 0.45 ? 1 - p / 0.45 : (p - 0.45) / 0.55;
      }
    }
    for (const g of eyes) {
      const [cx, cy] = at(g);
      g.setAttribute("transform",
        `translate(${cx} ${cy}) scale(1 ${Math.max(0.02, lid)}) translate(${-cx} ${-cy})`);
    }

    if (mouth) {
      const [cx, cy] = at(mouth);
      // 原画の口は 3x2px の点が 2 つきりなので、控えめに動かしても目に
      // 見えない。縦を大きく伸ばして、点が縦長になることで開閉に見せる。
      const wave = Math.sin(t * 1.3) * 0.5 + 0.5;            // 0..1
      const open = MOUTH_CLOSED + wave * (MOUTH_OPEN - MOUTH_CLOSED);
      mouth.setAttribute("transform",
        `translate(${cx} ${cy}) scale(1 ${open}) translate(${-cx} ${-cy})`);
    }

    if (ahoge) {
      const [cx, cy] = at(ahoge);
      // 根元を軸に揺らす。重心ではなく下端を中心にしたいので少し下げる
      const pivotY = cy + 26;
      const a = Math.sin(t * 1.5) * 3.2 + Math.sin(t * 2.7) * 1.1;
      ahoge.setAttribute("transform", `rotate(${a} ${cx} ${pivotY})`);
    }

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function render() {
  const stage = document.getElementById("stage");
  stage.innerHTML = "";
  const svg = build();
  stage.appendChild(svg);
  animate(svg);
  reportStats();
}

function reportStats() {
  const shapes = window.OKOJO_SHAPES || [];
  const counts = {};
  for (const s of shapes) counts[s.label] = (counts[s.label] || 0) + 1;
  const info = document.getElementById("info");
  if (info) {
    info.textContent =
      `${shapes.length} parts — ` +
      Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .map(([k, v]) => `${k}:${v}`)
        .join("  ");
  }
}

document.addEventListener("DOMContentLoaded", render);
