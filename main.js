// main.js
// shapes.js (tools/analyze.py が assets/SD.png を解析して生成した座標データ) から
// SVG を組み立てて描画する。実行時に画像は一切読み込まない。
//
// パーツは推定ラベル(hair / skin / iris / hoodie / legs ...)ごとに <g> へまとめ、
// Live2D 的に部位単位でアニメーションできるようにしている。

const NS = "http://www.w3.org/2000/svg";

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

    svg.appendChild(
      el("path", {
        id: s.id,
        "data-label": s.label,
        d: s.d,
        fill: paint,
        "fill-rule": "evenodd",
        // 隣接パーツの継ぎ目に地の色が出ないよう、同じ塗りで細く縁取る
        stroke: paint,
        "stroke-width": "0.8",
        "stroke-linejoin": "round",
      })
    );
  }

  return svg;
}

function render() {
  const stage = document.getElementById("stage");
  stage.innerHTML = "";
  stage.appendChild(build());
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
