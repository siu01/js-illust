// main.js
// shapes.js (tools/analyze.py が assets/SD.png を解析して生成した座標データ) から
// SVG を組み立てて描画する。実行時に画像は一切読み込まない。
//
// パーツは推定ラベル(hair / skin / iris / hoodie / legs ...)ごとに <g> へまとめ、
// Live2D 的に部位単位でアニメーションできるようにしている。

const NS = "http://www.w3.org/2000/svg";

// 目の領域の半幅・半高 (viewBox)。この矩形に描画範囲ごと収まるパーツを
// 「その目のもの」とみなす。片目の幅はおよそ 50。
const EYE_HALF_W = 27;
const EYE_HALF_H = 23;

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
      "stroke-width": "1.1",
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
  const eyes = all.filter((s) => s.label === "eye");
  if (!eyes.length) return [];

  const median = (xs) => {
    const v = [...xs].sort((a, b) => a - b);
    return v[Math.floor(v.length / 2)];
  };

  // 左右に分けたうえで、それぞれの中心を中央値で決める。平均だと、判定に
  // 紛れ込んだ外れ値ひとつで中心がずれる。
  const mid = median(eyes.map((s) => s.cx));
  const sides = [eyes.filter((s) => s.cx < mid), eyes.filter((s) => s.cx >= mid)];

  const byId = new Map(all.map((s) => [s.id, s]));

  const anchors = window.OKOJO_ANCHORS || {};
  const halfW = anchors.eyeHalfW || EYE_HALF_W;
  const halfH = anchors.eyeHalfH || EYE_HALF_H;

  return sides.filter((side) => side.length).map((side, i) => {
    // 中心は解析側が瞳から出したものを使う。こちらでラベルの重心を取ると
    // こめかみ側のパーツに引っ張られ、虹彩が範囲から外れてしまう。
    const a = anchors["eye" + (i ? "R" : "L")];
    const cx = a ? a.cx : median(side.map((s) => s.cx));
    const cy = a ? a.cy : median(side.map((s) => s.cy));
    const x0 = cx - halfW, x1 = cx + halfW;
    const y0 = cy - halfH, y1 = cy + halfH;

    // 目の領域に「描画範囲ごと」収まるものだけを集める。重心で判定すると、
    // 目の上を通り過ぎるだけの長い髪の線まで入り、グループの外形が顔ごと
    // 覆う大きさに化ける。この条件なら白目もまつ毛も入り、通過する線は入らない。
    const members = [];
    for (const s of all) {
      if (s.label === "mouth" || s.label === "ahoge") continue;
      const n = svg.getElementById(s.id);
      if (!n) continue;
      const b = n.getBBox();
      if (b.x >= x0 && b.x + b.width <= x1 && b.y >= y0 && b.y + b.height <= y1) {
        members.push(n);
      }
    }
    if (!members.length) return null;

    const g = el("g", { "data-part": "eye" + (i ? "R" : "L") });
    members[members.length - 1].after(g);
    // 取り込みで重なり順が崩れるので、解析時の depth で並べ直す
    members.sort((a, z) => (byId.get(a.id)?.depth ?? 0) - (byId.get(z.id)?.depth ?? 0));
    for (const n of members) g.appendChild(n);

    g.dataset.cx = cx;
    g.dataset.cy = cy;
    return g;
  }).filter(Boolean);
}

// ------------------------------------------------------------ アニメーション
function animate(svg) {
  const eyes = groupEyes(svg);
  const mouth = groupParts(svg, "mouth");
  const inner = groupParts(svg, "mouthInner");
  const ahoge = groupParts(svg, "ahoge");
  if (!eyes.length && !mouth && !ahoge) return;

  const at = (g) => [parseFloat(g.dataset.cx), parseFloat(g.dataset.cy)];

  // 口内は上端を軸に開かせたいので、潰す前の外形から上端を取っておく
  const innerTop = inner ? inner.getBBox().y : 0;

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

    // 口の開き具合。0 = 閉じ、1 = 全開
    const openness = Math.sin(t * 1.3) * 0.5 + 0.5;

    if (mouth) {
      const [cx, cy] = at(mouth);
      // 原画の口は 3x2px の点が 2 つきりなので、控えめに動かしても目に
      // 見えない。縦を大きく伸ばして、点が縦長になることで開閉に見せる。
      const open = MOUTH_CLOSED + openness * (MOUTH_OPEN - MOUTH_CLOSED);
      mouth.setAttribute("transform",
        `translate(${cx} ${cy}) scale(1 ${open}) translate(${-cx} ${-cy})`);
    }

    if (inner) {
      // 上端を固定して下に開く。中心を固定すると、口内が唇より上にも
      // 広がって顔にめり込む。
      const [cx] = at(inner);
      const top = innerTop;
      // 閉じている間は完全に潰す。細く残すと、閉じた口の隙間から暗い線が覗く
      const h = openness * openness;                 // 開きはじめを緩やかに
      inner.setAttribute("transform",
        `translate(${cx} ${top}) scale(1 ${Math.max(0.001, h)}) translate(${-cx} ${-top})`);
      inner.setAttribute("opacity", h < 0.04 ? 0 : 1);
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
