// STrace Inspector
// Copyright (C) 2024 Forschungszentrum Juelich GmbH, Juelich Supercomputing Centre

// Contributors:
// - Aravind Sankaran

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.

// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

function quantile(s, q) {
    const pos = (s.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    return s[base + 1] !== undefined
    ? s[base] + rest * (s[base + 1] - s[base])
    : s[base];
}

function filter_outliers(values){
    const q1 = quantile(values, 0.25);
    const q3 = quantile(values, 0.75);  
    const iqr = q3 - q1;
    const lowerFence = q1 - 1.5 * iqr;
    const upperFence = q3 + 1.5 * iqr;
    const inliers = values.filter(v => v >= lowerFence && v <= upperFence);
    const outliers = values.filter(v => v < lowerFence || v > upperFence);
    return {
        inliers: inliers,
        outliers: outliers
    }
}

// Removes outliers (1.5 IQR), then computes Q1/median/Q3 on cleaned data.
function computeBoxStats(values, removeOutliers = false) {
    let s0 = [...values].sort((a,b)=>a - b);
    let outliers = [];
    
    if (removeOutliers){
    let v = filter_outliers(s0);
    s0 = v.inliers;
    outliers = v.outliers;
    } 

    if (s0.length === 0) {
    return {
        q1: NaN, median: NaN, q3: NaN,
        whiskerMin: NaN, whiskerMax: NaN,
        inliers: [], outliers: []
    };
    }

    const q1 = quantile(s0, 0.25);
    const median = quantile(s0, 0.5);
    const q3 = quantile(s0, 0.75);
    const iqr = q3 - q1;
    const lower = q1 - 1.5 * iqr;
    const upper = q3 + 1.5 * iqr;

    let whiskerMin = lower;
    let whiskerMax = upper;

    if (lower <= s0[0]) {
    whiskerMin = s0[0];
    }
    if (upper >= s0[s0.length - 1]) {
    whiskerMax = s0[s0.length - 1];
    }

    return { q1, median, q3, whiskerMin, whiskerMax, inliers: s0, outliers: outliers };
}

function renderPRBoxplot(data, options = {}) {
  const showOutliers = options.showOutliers !== false;  // default true
  const xlabel = options.xlabel || null;

  // New configurable sizes
  const rowHeight = options.rowHeight || 40;     // default row height
  const xTickSize = options.xTickSize || 14;     // default x tick font size
  const yTickSize = options.yTickSize || 14;     // default y tick font size
  const xlabelSize = options.xlabelSize || 16;   // default x label font size
  const width = options.width || 1000;            // default width
  const svgId = options.svgId || '#d3SVG'; // default SVG element ID
  const ranks = options.rank || null; // optional ranks for y-axis ordering

  let labels = Object.keys(data);
  if (ranks) {
    labels = Object.keys(ranks).sort((a, b) => ranks[a] - ranks[b]);
  }
  const stats = labels.map(k => computeBoxStats(data[k]));

  const svg = d3.select(svgId);
  console.log(svg);
  svg.selectAll('*').remove(); // clear previous

  const margin = { top: 40, right: 1, bottom: 60, left: 100 };

  // Dynamic height based on how many labels
  const height = margin.top + margin.bottom + labels.length * rowHeight;

  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const g = svg
    .attr('width', width)
    .attr('height', height)
    .append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  const allValues = stats.flatMap(s => [...s.inliers, ...(showOutliers ? s.outliers : [])]);
  if (allValues.length === 0) return;

  let [minVal, maxVal] = d3.extent(allValues);
  const range = maxVal - minVal || 1;
  minVal = minVal - 0.05 * range;
  maxVal = maxVal + 0.05 * range;

  const x = d3.scaleLinear()
    .domain([minVal, maxVal])
    .range([0, innerW]);

  const y = d3.scaleBand()
    .domain(labels)
    .range([0, innerH])
    .padding(0.4);

  // ----- Axes -----

  // Y-axis
  const yAxis = g.append('g')
    .call(d3.axisLeft(y));

  yAxis.selectAll("text")
    .style("font-size", yTickSize + "px");
  yAxis.select(".domain").remove();

  // X-axis
  const xAxis = g.append('g')
    .attr('transform', `translate(0,${innerH})`)
    .call(d3.axisBottom(x));

  xAxis.selectAll("text")
    .style("font-size", xTickSize + "px");
  xAxis.select(".domain").remove();

  // Draw axis bounding box (like Matplotlib)
  g.append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", innerW)
    .attr("height", innerH)
    .attr("fill", "none")
    .attr("stroke", "black")
    .attr("stroke-width", 1);

  // ----- X label -----
  if (xlabel) {
    svg.append("text")
      .attr("class", "x label")
      .attr("text-anchor", "middle")
      .attr("x", margin.left + innerW / 2)
      .attr("y", height - 10)
      .style("font-size", xlabelSize + "px")
      .text(xlabel);
  }

  // ----- Draw boxplots -----
  labels.forEach((label, i) => {
    const s = stats[i];
    if (!isFinite(s.q1) || !isFinite(s.q3)) return;

    const cy = y(label) + y.bandwidth() / 2;
    const boxH = y.bandwidth() * 0.6;

    // Whiskers
    g.append('line')
      .attr('class', 'whisker')
      .attr('stroke', 'purple')
      .attr('stroke-width', 1.4)
      .attr('stroke-dasharray', '4 3')
      .attr('x1', x(s.whiskerMin)).attr('x2', x(s.q1))
      .attr('y1', cy).attr('y2', cy);

    g.append('line')
      .attr('class', 'whisker')
      .attr('stroke', 'purple')
      .attr('stroke-width', 1.4)
      .attr('stroke-dasharray', '4 3')
      .attr('x1', x(s.whiskerMax)).attr('x2', x(s.q3))
      .attr('y1', cy).attr('y2', cy);

    // Whisker caps
    g.append('line')
      .attr('class', 'whisker')
      .attr('stroke', 'purple')
      .attr('stroke-width', 1.4)
      .attr('stroke-dasharray', '4 3')
      .attr('x1', x(s.whiskerMin)).attr('x2', x(s.whiskerMin))
      .attr('y1', cy - boxH / 3).attr('y2', cy + boxH / 3);

    g.append('line')
      .attr('class', 'whisker')
      .attr('stroke', 'purple')
      .attr('stroke-width', 1.4)
      .attr('stroke-dasharray', '4 3')
      .attr('x1', x(s.whiskerMax)).attr('x2', x(s.whiskerMax))
      .attr('y1', cy - boxH / 3).attr('y2', cy + boxH / 3);

    // Box
    g.append('rect')
      .attr('class', 'box')
      .attr('fill', 'lightgray')
      .attr('stroke', 'black')
      .attr('stroke-width', 1.4)
      .attr('x', x(s.q1))
      .attr('y', cy - boxH / 2)
      .attr('width', x(s.q3) - x(s.q1))
      .attr('height', boxH);

    // Median
    g.append('line')
      .attr('class', 'median')
      .attr('stroke', 'red')
      .attr('stroke-width', 2)
      .attr('x1', x(s.median))
      .attr('x2', x(s.median))
      .attr('y1', cy - boxH / 2)
      .attr('y2', cy + boxH / 2);

    // Points
    // const allPoints = [...s.inliers, ...s.outliers];
    const allPoints = [...s.outliers];

    g.selectAll(`.point-${i}`)
      .data(allPoints)
      .enter()
      .append('circle')
      .attr('class', 'point')
      .attr('fill', 'blue')
      .attr('cx', d => x(d))
      .attr('cy', cy)
      .attr('r', 3);
  });
}


function saveSvg(svgElement, filename = "plot.svg") {
  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(svgElement);

  // Add XML header
  if (!source.match(/^<svg[^>]+xmlns="http:\/\/www\.w3\.org\/2000\/svg"/)) {
    source = source.replace(
      /^<svg/,
      '<svg xmlns="http://www.w3.org/2000/svg"'
    );
  }

  // Make XML pretty
  source = '<?xml version="1.0" standalone="no"?>\n' + source;

  const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

function addZoomToolbar(container, chart) {
  container.style.position = "relative";

  const toolbar = document.createElement("div");
  toolbar.style.position = "absolute";
  // initial values; will be corrected after we know canvas position
  toolbar.style.top = "0px";
  toolbar.style.right = "0px";
  toolbar.style.display = "flex";
  toolbar.style.alignItems = "center";
  toolbar.style.gap = "10px";
  toolbar.style.padding = "6px 10px";
  toolbar.style.background = "rgba(255,255,255,0.9)";
  toolbar.style.borderRadius = "6px";
  toolbar.style.boxShadow = "0 2px 6px rgba(0,0,0,0.15)";
  toolbar.style.opacity = "0";
  toolbar.style.transition = "opacity 0.25s ease";
  toolbar.style.pointerEvents = "none";
  toolbar.style.zIndex = "20";

  // Fade in/out on hover
  container.addEventListener("mouseenter", () => {
    toolbar.style.opacity = "1";
    toolbar.style.pointerEvents = "auto";
  });
  container.addEventListener("mouseleave", () => {
    toolbar.style.opacity = "0";
    toolbar.style.pointerEvents = "none";
  });

  // Save button (Font Awesome download icon)
  const saveBtn = document.createElement("div");
  saveBtn.innerHTML = `<i class="fa-solid fa-download"></i>`;
  saveBtn.style.cursor = "pointer";
  saveBtn.style.fontSize = "18px";
  saveBtn.style.padding = "4px";
  saveBtn.style.borderRadius = "4px";

  saveBtn.onmouseenter = () => saveBtn.style.background = "rgba(0,0,0,0.07)";
  saveBtn.onmouseleave = () => saveBtn.style.background = "transparent";

  saveBtn.onclick = () => {
    const link = document.createElement("a");
    link.href = chart.toBase64Image();
    link.download = "chart.png";
    link.click();
  };

  // Info text
  const info = document.createElement("div");
  info.textContent = "scroll to zoom";
  info.style.fontSize = "12px";
  info.style.color = "#777";

  toolbar.appendChild(info);
  toolbar.appendChild(saveBtn);
  container.appendChild(toolbar);

  // --- NEW: position toolbar at the top-right of the canvas inside the container ---
  function positionToolbar() {
    const canvas = container.querySelector("canvas");
    if (!canvas) return;

    const containerRect = container.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();

    const topOffset = canvasRect.top - containerRect.top;
    const rightOffset = containerRect.right - canvasRect.right;

    toolbar.style.top = (topOffset + 8) + "px";
    toolbar.style.right = (rightOffset + 8) + "px";
  }

  positionToolbar();
  window.addEventListener("resize", positionToolbar);
}


function renderPRBoxplotChartJS(data, options = {}) {
  const showOutliers = options.showOutliers !== false;
  const xlabel = options.xlabel || null;

  const rowHeight = options.rowHeight || 40;
  const xTickSize = options.xTickSize || 14;
  const yTickSize = options.yTickSize || 14;
  const xlabelSize = options.xlabelSize || 16;

  const width = options.width || 1000;
  const containerOpt = options.container || "#prBoxplotContainer";

  const container = typeof containerOpt === "string"
    ? document.querySelector(containerOpt)
    : containerOpt;

  if (!container) return;

  const ranks = options.rank || null; // optional ranks for y-axis ordering

  let labels = Object.keys(data);
  if (ranks) {
    labels = Object.keys(ranks).sort((a, b) => ranks[a] - ranks[b]);
  }

  // IMPORTANT: compute stats in the correct order as labels
  const stats = labels.map(k => computeBoxStats(data[k]));

  if (ranks){
    labels = labels.map(l => `${l} (${ranks[l]})`);
  }


  const allValues = stats.flatMap(s => [
    ...s.inliers,
    ...(showOutliers ? s.outliers : [])
  ]);

  if (allValues.length === 0) return;

  let minVal = Math.min(...allValues);
  let maxVal = Math.max(...allValues);
  const range = maxVal - minVal || 1;
  minVal -= 0.05 * range;
  maxVal += 0.05 * range;

  const height = 40 + labels.length * rowHeight;

  container.innerHTML = "";
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  container.appendChild(canvas);

  const ctx = canvas.getContext("2d");

  const chartData = {
    labels,
    datasets: [{
      label: "Boxplot",
      data: stats.map(s => s.median),
      backgroundColor: "transparent"
    }]
  };

  const boxplotPlugin = {
    id: "customBoxplot",
    afterDatasetDraw(chart, args) {
      const { ctx, chartArea, scales } = chart;
      const xScale = scales.x;
      const meta = chart.getDatasetMeta(args.index);

      // --- FIX: clip to chart area ---
      ctx.save();
      ctx.beginPath();
      ctx.rect(chartArea.left, chartArea.top,
              chartArea.right - chartArea.left,
              chartArea.bottom - chartArea.top);
      ctx.clip();

      stats.forEach((s, i) => {
        if (!isFinite(s.q1) || !isFinite(s.q3)) return;
        const bar = meta.data[i];
        if (!bar) return;

        const cy = bar.y;
        const barH = bar.height || 20;
        const boxH = barH * 0.6;

        const xQ1 = xScale.getPixelForValue(s.q1);
        const xQ3 = xScale.getPixelForValue(s.q3);
        const xMed = xScale.getPixelForValue(s.median);
        const xWMin = xScale.getPixelForValue(s.whiskerMin);
        const xWMax = xScale.getPixelForValue(s.whiskerMax);

        // Whiskers
        ctx.strokeStyle = "purple";
        ctx.lineWidth = 1.4;
        ctx.setLineDash([4, 3]);

        ctx.beginPath();
        ctx.moveTo(xWMin, cy);
        ctx.lineTo(xQ1, cy);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(xWMax, cy);
        ctx.lineTo(xQ3, cy);
        ctx.stroke();

        // Whisker caps
        ctx.beginPath();
        ctx.moveTo(xWMin, cy - boxH/3);
        ctx.lineTo(xWMin, cy + boxH/3);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(xWMax, cy - boxH/3);
        ctx.lineTo(xWMax, cy + boxH/3);
        ctx.stroke();

        ctx.setLineDash([]);

        // Box rectangle
        ctx.fillStyle = "lightgray";
        ctx.strokeStyle = "black";
        ctx.lineWidth = 1.4;

        ctx.beginPath();
        ctx.rect(xQ1, cy - boxH/2, xQ3 - xQ1, boxH);
        ctx.fill();
        ctx.stroke();

        // Median line
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(xMed, cy - boxH/2);
        ctx.lineTo(xMed, cy + boxH/2);
        ctx.stroke();

        // Outliers
        ctx.fillStyle = "blue";
        s.outliers.forEach(val => {
          const px = xScale.getPixelForValue(val);
          ctx.beginPath();
          ctx.arc(px, cy, 3, 0, Math.PI * 2);
          ctx.fill();
        });

        // Inliers (optional)
        
        ctx.fillStyle = "blue";
        s.inliers.forEach(val => {
          const px = xScale.getPixelForValue(val);
          ctx.beginPath();
          ctx.arc(px, cy, 2, 0, Math.PI * 2);
          ctx.fill();
        });
        
      });

      // --- STOP clipping ---
      ctx.restore();

      // Draw outer border after clipping
      ctx.save();
      ctx.strokeStyle = "black";
      ctx.strokeRect(
        chartArea.left,
        chartArea.top,
        chartArea.right - chartArea.left,
        chartArea.bottom - chartArea.top
      );
      ctx.restore();
    }
  };

  const myChart = new Chart(ctx, {
    type: "bar",
    data: chartData,
    options: {
      responsive: false,
      indexAxis: "y",
      scales: {
        x: {
          min: minVal,
          max: maxVal,
          ticks: { font: { size: xTickSize } },
          title: xlabel ? {
            display: true,
            text: xlabel,
            font: { size: xlabelSize }
          } : undefined
        },
        y: {
          ticks: { font: { size: yTickSize } }
        }
      },
      plugins: {
        legend: { display: false },

        // 🔥🔥🔥 ZOOM ENABLED HERE
        zoom: {
          pan: {
            enabled: true,
            mode: "x"
          },
          zoom: {
            wheel: {
              enabled: true
            },
            pinch: {
              enabled: true
            },
            mode: "x"
          },
          limits: {
            x: { min: minVal, max: maxVal }
          }
        }
      }
    },
    plugins: [boxplotPlugin]
  });

  addZoomToolbar(container, myChart);
}