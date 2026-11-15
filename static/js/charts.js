/**
 * charts.js — Plotly.js chart renderers
 * "CedarField-inspired" palette: #FF4500, #D9381E, #0F0F0F, #171717,
 *                                    #D4D4D4, #FFFFFF, #FF3355
 */

const Charts = (() => {

  /* ── Shared Plotly theme ─────────────────────────────────── */
  const C = {
    bg:      '#0F0F0F',
    surface: '#171717',
    accent:  '#FF4500',
    rust:    '#D9381E',
    danger:  '#FF3355',
    text:    '#FFFFFF',
    sub:     '#D4D4D4',
    muted:   '#8A8A8A',
    healthy: '#A3A3A3',
    failure: '#FF4500',
    green:   '#4CAF76',
  };

  const BASE_LAYOUT = {
    paper_bgcolor: C.surface,
    plot_bgcolor:  C.bg,
    font: { color: C.text, family: "'Inter', sans-serif", size: 11 },
    margin: { l: 60, r: 20, t: 44, b: 50 },
    xaxis: {
      gridcolor: 'rgba(212,212,212,0.08)',
      zerolinecolor: 'rgba(212,212,212,0.15)',
      tickfont: { color: C.sub },
    },
    yaxis: {
      gridcolor: 'rgba(212,212,212,0.08)',
      zerolinecolor: 'rgba(212,212,212,0.15)',
      tickfont: { color: C.sub },
    },
    legend: {
      bgcolor: 'rgba(23,23,23,0.85)',
      bordercolor: 'rgba(255,69,0,0.25)',
      borderwidth: 1,
      font: { color: C.sub, size: 10 },
    },
  };

  const CFG = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['select2d','lasso2d','autoScale2d','resetScale2d'],
    toImageButtonOptions: { format: 'png', scale: 2, filename: 'pred_maint' },
  };

  function layout(overrides = {}) {
    return Plotly.relayout ? { ...BASE_LAYOUT, ...overrides } : Object.assign({}, BASE_LAYOUT, overrides);
  }

  function plot(id, traces, lay, cfg = CFG) {
    Plotly.newPlot(id, traces, Object.assign({}, BASE_LAYOUT, lay), cfg);
  }

  /* ── Helper: stat HTML ───────────────────────────────────── */
  function statCards(items) {
    return `<div class="stat-cards-row">${items.map(i =>
      `<div class="stat-card">
        <div class="stat-val ${i.cls||''}">${i.val}</div>
        <div class="stat-label">${i.label}</div>
      </div>`
    ).join('')}</div>`;
  }

  function tableHtml(headers, rows, cls = '') {
    return `<div class="table-wrap"><table class="data-table ${cls}">
      <thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead>
      <tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody>
    </table></div>`;
  }

  /* ══════════════════════════════════════════════════════════
     SECTION 2 — DATA LOAD
  ══════════════════════════════════════════════════════════ */
  function renderLoad(data, containerId) {
    const el = document.getElementById(containerId);
    const cb = data.class_balance;

    el.innerHTML = `
      ${statCards([
        { val: data.shape.rows.toLocaleString(), label: 'Total Rows' },
        { val: data.shape.cols,                 label: 'Columns' },
        { val: data.missing,                    label: 'Missing Values' },
        { val: data.duplicates,                 label: 'Duplicate Rows' },
        { val: `${cb.failure_pct}%`,            label: 'Failure Rate', cls: 'text-danger' },
        { val: data.cached ? 'Cached' : 'Downloaded', label: 'Data Source' },
      ])}
      <div class="chart-grid-2">
        <div class="chart-card">
          <div class="chart-card-title">Class Balance</div>
          <div class="chart-container" id="ch-class-bal" style="height:280px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-card-title">Column Roles</div>
          <div class="chart-container" id="ch-col-roles" style="height:280px"></div>
        </div>
      </div>
      ${tableHtml(
        ['Column','Type','Role','Non-Null','Unique','Sample'],
        data.dtypes.map(d => [
          `<code>${d.column}</code>`,
          d.dtype,
          `<span class="badge badge-${d.role==='Target'?'target':d.role==='Leakage'?'leak':d.role==='ID'?'id':'feat'}">${d.role}</span>`,
          d.non_null,
          d.unique,
          `<code>${d.sample}</code>`,
        ])
      )}`;

    // Class balance donut
    plot('ch-class-bal', [{
      type: 'pie',
      hole: 0.65,
      values: [cb.healthy_count, cb.failure_count],
      labels: ['Healthy (0)', 'Failure (1)'],
      marker: { colors: [C.healthy, C.danger], line: { color: C.surface, width: 2 } },
      textinfo: 'label+percent',
      textfont: { color: C.text, size: 11 },
      hovertemplate: '<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>',
    }], {
      showlegend: false,
      margin: { l: 10, r: 10, t: 20, b: 10 },
      annotations: [{
        text: `<b>${cb.failure_pct}%</b><br><span style="font-size:10px">Failure</span>`,
        x: 0.5, y: 0.5, font: { color: C.danger, size: 16 },
        showarrow: false,
      }],
    });

    // Role breakdown pie
    const roleCounts = {};
    data.dtypes.forEach(d => { roleCounts[d.role] = (roleCounts[d.role] || 0) + 1; });
    const roleColors = { Feature: C.accent, Target: C.green, Leakage: C.danger, ID: C.muted };
    plot('ch-col-roles', [{
      type: 'pie',
      hole: 0.5,
      values: Object.values(roleCounts),
      labels: Object.keys(roleCounts),
      marker: { colors: Object.keys(roleCounts).map(r => roleColors[r] || C.sub) },
      textinfo: 'label+value',
      textfont: { color: C.text },
    }], { showlegend: true, margin: { l: 10, r: 10, t: 20, b: 10 } });
  }

  /* ══════════════════════════════════════════════════════════
     SECTION 3 — PREPROCESS
  ══════════════════════════════════════════════════════════ */
  function renderPreprocess(data, containerId) {
    const el = document.getElementById(containerId);

    // Stats table rows
    const statsRows = data.stats.map(s => [
      `<code style="color:${s.is_engineered ? C.accent : C.sub}">${s.feature}</code>`,
      s.min.toFixed(2),
      s.mean.toFixed(2),
      s.median.toFixed(2),
      s.max.toFixed(2),
      s.std.toFixed(2),
    ]);

    el.innerHTML = `
      ${statCards([
        { val: data.columns_removed.length, label: 'Columns Removed', cls: 'text-danger' },
        { val: data.new_features.length,    label: 'Features Engineered', cls: 'text-green' },
        { val: `${data.shape.rows.toLocaleString()} × ${data.shape.cols}`, label: 'Clean Shape' },
      ])}
      <div class="section-divider">Removed Columns (Leakage + IDs)</div>
      <div class="chips-wrap">${data.columns_removed.map(c => `<span class="chip">${c}</span>`).join('')}</div>
      <div class="section-divider">Summary Statistics — <span style="color:${C.accent}">Amber = Engineered Feature</span></div>
      ${tableHtml(['Feature','Min','Mean','Median','Max','Std'], statsRows)}
      <div id="ch-feat-compare" style="height:300px"></div>`;

    // Bar chart: std / max comparison for engineered vs raw
    const eng   = data.stats.filter(s =>  s.is_engineered);
    const raw   = data.stats.filter(s => !s.is_engineered);
    plot('ch-feat-compare', [
      { name: 'Raw Features',      type: 'bar', x: raw.map(s => s.feature), y: raw.map(s => s.std), marker: { color: C.healthy, opacity: 0.8 } },
      { name: 'Engineered Features', type: 'bar', x: eng.map(s => s.feature), y: eng.map(s => s.std), marker: { color: C.accent, opacity: 0.9 } },
    ], {
      title: { text: 'Feature Standard Deviation', font: { color: C.sub, size: 12 } },
      barmode: 'group',
      xaxis: { title: 'Feature', tickangle: -30 },
      yaxis: { title: 'Std Dev' },
      margin: { l: 60, r: 20, t: 50, b: 100 },
    });
  }

  /* ══════════════════════════════════════════════════════════
     SECTION 4 — EDA
  ══════════════════════════════════════════════════════════ */
  function renderEDA(data, containerId) {
    const el = document.getElementById(containerId);
    const bd = data.box_data;

    el.innerHTML = `
      <div class="chart-grid-2">
        <div class="chart-card"><div class="chart-card-title">Class Imbalance</div><div class="chart-container" id="ch-eda-bal" style="height:280px"></div></div>
        <div class="chart-card"><div class="chart-card-title">Failure Sub-Types</div><div class="chart-container" id="ch-eda-modes" style="height:280px"></div></div>
      </div>
      <div class="chart-grid-3">
        <div class="chart-card"><div class="chart-card-title">Mechanical Power (W)</div><div class="chart-container" id="ch-box-power" style="height:260px"></div></div>
        <div class="chart-card"><div class="chart-card-title">Overstrain Index</div><div class="chart-container" id="ch-box-overstrain" style="height:260px"></div></div>
        <div class="chart-card"><div class="chart-card-title">Temperature Diff (K)</div><div class="chart-container" id="ch-box-temp" style="height:260px"></div></div>
        <div class="chart-card"><div class="chart-card-title">Torque (Nm)</div><div class="chart-container" id="ch-box-torque" style="height:260px"></div></div>
        <div class="chart-card"><div class="chart-card-title">Rotational Speed (rpm)</div><div class="chart-container" id="ch-box-rpm" style="height:260px"></div></div>
        <div class="chart-card"><div class="chart-card-title">Tool Wear (min)</div><div class="chart-container" id="ch-box-wear" style="height:260px"></div></div>
      </div>
      <div class="chart-card mb-20">
        <div class="chart-card-title">Correlation Matrix (all features)</div>
        <div class="chart-container" id="ch-corr" style="height:400px"></div>
      </div>
      <div class="chart-card">
        <div class="chart-card-title">Machine Type vs Failure Rate</div>
        <div class="chart-container" id="ch-type" style="height:280px"></div>
      </div>`;

    // Class balance donut
    plot('ch-eda-bal', [{
      type: 'pie', hole: 0.6,
      values: data.class_balance.counts,
      labels: data.class_balance.labels,
      marker: { colors: [C.healthy, C.danger] },
      textinfo: 'label+percent',
      textfont: { color: C.text },
      hovertemplate: '<b>%{label}</b><br>%{value:,} samples<extra></extra>',
    }], { showlegend: false, margin: { l:5,r:5,t:10,b:5 } });

    // Failure sub-modes bar
    plot('ch-eda-modes', [{
      type: 'bar',
      x: data.failure_modes.labels,
      y: data.failure_modes.counts,
      marker: {
        color: [C.danger, C.rust, C.accent, C.healthy, C.muted],
        line: { color: 'rgba(0,0,0,0.3)', width: 1 },
      },
      text: data.failure_modes.counts,
      textposition: 'outside',
      textfont: { color: C.text },
      hovertemplate: '<b>%{x}</b><br>%{y} failures<extra></extra>',
    }], {
      yaxis: { title: 'Count' },
      margin: { l:50,r:10,t:10,b:40 },
      showlegend: false,
    });

    // Box plots
    const boxSpec = [
      ['ch-box-power',     'Power_W'],
      ['ch-box-overstrain','Overstrain_Index'],
      ['ch-box-temp',      'Temp_Diff_K'],
      ['ch-box-torque',    'Torque [Nm]'],
      ['ch-box-rpm',       'Rotational speed [rpm]'],
      ['ch-box-wear',      'Tool wear [min]'],
    ];
    boxSpec.forEach(([divId, feat]) => {
      if (!bd[feat]) return;
      const { healthy, failure, unit } = bd[feat];
      plot(divId, [
        { type:'box', y: healthy, name:'Healthy', marker:{color: C.healthy}, boxmean:'sd', jitter: 0, whiskerwidth: 0.4,
          hovertemplate: '<b>Healthy</b><br>%{y:.2f} ' + unit + '<extra></extra>' },
        { type:'box', y: failure, name:'Failure', marker:{color: C.danger},  boxmean:'sd', jitter: 0, whiskerwidth: 0.4,
          hovertemplate: '<b>Failure</b><br>%{y:.2f} ' + unit + '<extra></extra>' },
      ], {
        yaxis: { title: unit },
        margin: { l:50,r:10,t:10,b:40 },
        showlegend: true,
        legend: { orientation: 'h', y: -0.15 },
      });
    });

    // Correlation heatmap
    const corr = data.correlation;
    plot('ch-corr', [{
      type: 'heatmap',
      z:    corr.z,
      x:    corr.x,
      y:    corr.y,
      text: corr.text,
      texttemplate: '%{text}',
      colorscale: [
        [0,   C.danger],
        [0.25,'#3D0A14'],
        [0.5,  C.bg],
        [0.75,'#7A2408'],
        [1,   C.accent],
      ],
      zmin: -1, zmax: 1,
      hovertemplate: '<b>%{y} × %{x}</b><br>r = %{z:.2f}<extra></extra>',
      colorbar: {
        tickfont: { color: C.sub },
        outlinecolor: 'transparent',
      },
    }], {
      margin: { l:80, r:10, t:10, b:80 },
      xaxis: { tickangle: -40, tickfont:{color: C.sub, size:10} },
      yaxis: { tickfont:{color: C.sub, size:10} },
    });

    // Type distribution stacked bar
    const td = data.type_distribution;
    plot('ch-type', [
      { type:'bar', name:'Healthy', x: td.types, y: td.healthy, marker:{color: C.healthy, opacity:0.8} },
      { type:'bar', name:'Failure', x: td.types, y: td.failure, marker:{color: C.danger,  opacity:0.9},
        text: td.failure_rates.map(r=>`${r}%`), textposition:'outside', textfont:{color:C.danger} },
    ], {
      barmode: 'stack',
      yaxis: { title: 'Machine Cycles' },
      xaxis: { title: 'Product Type (L / M / H)' },
      margin: { l:60,r:10,t:10,b:50 },
    });
  }

  /* ══════════════════════════════════════════════════════════
     SECTION 5 — SPLIT
  ══════════════════════════════════════════════════════════ */
  function renderSplit(data, containerId) {
    const el = document.getElementById(containerId);
    const { train, val, test } = data;

    el.innerHTML = `
      ${statCards([
        { val: data.total.toLocaleString(), label: 'Total Samples' },
        { val: `${train.rows.toLocaleString()} (70%)`, label: 'Train Set' },
        { val: `${val.rows.toLocaleString()} (15%)`,   label: 'Validation Set' },
        { val: `${test.rows.toLocaleString()} (15%)`,  label: 'Test Set (Locked)' },
      ])}
      <div class="split-visual">
        <div class="split-seg" style="flex:70;background:linear-gradient(135deg,#1A1B3A,#4338CA)">Train 70%</div>
        <div class="split-seg" style="flex:15;background:linear-gradient(135deg,#D9381E,#FF4500)">Val 15%</div>
        <div class="split-seg" style="flex:15;background:linear-gradient(135deg,#3A3A3A,#A3A3A3)">Test 15%</div>
      </div>
      <div id="ch-split-bar" style="height:300px"></div>
      <div class="section-divider">Stratification Check — Failure Rate per Split</div>
      ${tableHtml(
        ['Split','Rows','Failures','Failure %','Stratified?'],
        [
          ['Train',      train.rows.toLocaleString(), train.failures, `${train.failure_pct}%`, '✅ Yes'],
          ['Validation', val.rows.toLocaleString(),   val.failures,   `${val.failure_pct}%`,   '✅ Yes'],
          ['Test',       test.rows.toLocaleString(),  test.failures,  `${test.failure_pct}%`,  '✅ Yes'],
        ]
      )}
      <div class="section-divider">Features Used</div>
      <div class="chips-wrap">${data.num_features.map(f=>`<span class="chip" style="background:rgba(255,69,0,0.1);color:#FF4500;border-color:rgba(255,69,0,0.3)">${f}</span>`).join('')}
      ${data.cat_features.map(f=>`<span class="chip" style="background:rgba(129,140,248,0.15);color:#818CF8;border-color:rgba(129,140,248,0.3)">${f}</span>`).join('')}</div>`;

    plot('ch-split-bar', [
      { type:'bar', name:'Healthy',  x:['Train','Val','Test'], y:[train.rows - train.failures, val.rows - val.failures, test.rows - test.failures], marker:{color: C.healthy, opacity:0.8} },
      { type:'bar', name:'Failures', x:['Train','Val','Test'], y:[train.failures, val.failures, test.failures], marker:{color: C.danger} },
    ], {
      barmode: 'stack',
      yaxis: { title: 'Samples' },
      title: { text: 'Sample Distribution per Split', font:{color:C.sub,size:12} },
      margin: { l:60,r:20,t:50,b:50 },
    });
  }

  /* ══════════════════════════════════════════════════════════
     SECTION 6 — TRAINING
  ══════════════════════════════════════════════════════════ */
  function renderTrain(data, containerId) {
    const el = document.getElementById(containerId);
    const results = data.results;
    const models  = results.map(r => r.model);
    const metrics = ['cv_f1','val_precision','val_recall','val_f1','val_pr_auc'];
    const labels  = ['CV F1','Val Precision','Val Recall','Val F1','Val PR-AUC'];

    el.innerHTML = `
      <div id="ch-model-compare" style="height:360px"></div>
      ${tableHtml(
        ['Model','CV F1 (±std)','Val Precision','Val Recall','Val F1','Val PR-AUC'],
        results.map(r => [
          `<strong style="color:${r.color}">${r.model}</strong>`,
          `${r.cv_f1} ±${r.cv_f1_std}`,
          r.val_precision,
          r.val_recall,
          `<strong>${r.val_f1}</strong>`,
          r.val_pr_auc,
        ])
      )}`;

    const traces = metrics.map((m, mi) => ({
      type: 'bar',
      name: labels[mi],
      x:    models,
      y:    results.map(r => r[m]),
      text: results.map(r => r[m].toFixed(4)),
      textposition: 'outside',
      textfont: { color: C.text, size: 10 },
      marker: { color: [C.accent, C.rust, C.healthy, C.danger, C.sub][mi], opacity: 0.85 },
      hovertemplate: `<b>%{x}</b><br>${labels[mi]}: %{y:.4f}<extra></extra>`,
    }));

    plot('ch-model-compare', traces, {
      barmode: 'group',
      yaxis: { title: 'Score', range: [0, 1.1] },
      title: { text: 'Model Comparison — All Metrics', font:{color:C.sub,size:12} },
      margin: { l:60,r:20,t:50,b:80 },
      xaxis: { tickfont:{size:11} },
    });
  }

  /* ══════════════════════════════════════════════════════════
     SECTION 7 — TUNING
  ══════════════════════════════════════════════════════════ */
  function renderTune(data, containerId) {
    const el = document.getElementById(containerId);
    const bp = data.best_params;

    el.innerHTML = `
      <div class="best-params-card">
        <div class="bpc-title">🏆 Best Hyperparameters (5-Fold CV F1 = ${data.best_score})</div>
        <div class="bpc-params">
          ${Object.entries(bp).map(([k,v])=>`
            <div class="bpc-param">
              <span class="bpc-key">${k}</span>
              <span class="bpc-value">${v}</span>
            </div>`).join('')}
          <div class="bpc-param bpc-score">
            <span class="bpc-key">Best CV F1</span>
            <span class="bpc-value">${data.best_score}</span>
          </div>
        </div>
      </div>
      <div id="ch-grid-results" style="height:360px"></div>
      <div class="section-divider">All ${data.n_combinations} Combinations (sorted by CV F1)</div>
      ${tableHtml(
        ['Rank','n_estimators','max_depth','learning_rate','CV F1','Train F1'],
        data.grid_results.map((r,i) => [
          `#${i+1}`,
          r.n_estimators || '-',
          r.max_depth    || '-',
          r.learning_rate|| '-',
          `<strong style="color:${i===0?'#FF4500':C.sub}">${r.mean_test_score}</strong>`,
          r.mean_train_score || '-',
        ])
      )}`;

    const gr = data.grid_results;
    const labels = gr.map((r,i) => `#${i+1}: n=${r.n_estimators} d=${r.max_depth} lr=${r.learning_rate}`);
    plot('ch-grid-results', [
      {
        type: 'bar',
        x: labels,
        y: gr.map(r => r.mean_test_score),
        error_y: { type:'data', array: gr.map(r=>r.std_test_score||0), visible:true, color: C.muted },
        marker: {
          color: gr.map(r => r.mean_test_score),
          colorscale: [[0, C.rust],[0.5,C.accent2 || C.rust],[1, C.accent]],
          showscale: false,
        },
        text: gr.map(r => r.mean_test_score.toFixed(4)),
        textposition: 'outside',
        textfont: { color: C.text, size: 10 },
        hovertemplate: '<b>%{x}</b><br>CV F1: %{y:.4f}<extra></extra>',
      },
    ], {
      yaxis: { title: 'CV F1 Score', range:[0, Math.max(...gr.map(r=>r.mean_test_score))+0.05] },
      xaxis: { tickangle: -30, tickfont:{size:9} },
      title: { text: 'Grid Search — All Combinations', font:{color:C.sub,size:12} },
      margin: { l:60,r:20,t:50,b:140 },
    });
  }

  /* ══════════════════════════════════════════════════════════
     SECTION 8 — EVALUATION
  ══════════════════════════════════════════════════════════ */
  function renderEvaluate(data, containerId) {
    const el = document.getElementById(containerId);
    const cr = data.classification_report;
    const sm = data.summary;

    el.innerHTML = `
      ${statCards([
        { val: (cr.accuracy * 100).toFixed(2) + '%', label: 'Accuracy',   cls: 'text-green' },
        { val: cr.failure.precision.toFixed(4),       label: 'Failure Precision', cls: '' },
        { val: cr.failure.recall.toFixed(4),          label: 'Failure Recall',    cls: '' },
        { val: cr.failure['f1-score'].toFixed(4),     label: 'Failure F1',        cls: 'text-green' },
        { val: `${sm.tp} / ${sm.tp + sm.fn}`,         label: 'Failures Caught',   cls: 'text-green' },
        { val: sm.fp,                                  label: 'False Alarms',      cls: 'text-danger' },
      ])}
      <div class="chart-grid-2">
        <div class="chart-card">
          <div class="chart-card-title">Precision-Recall Curves (Test Set)</div>
          <div class="chart-container" id="ch-pr-curves" style="height:360px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-card-title">Confusion Matrix (Tuned Gradient Boosting)</div>
          <div class="chart-container" id="ch-conf-mat" style="height:360px"></div>
        </div>
      </div>
      <div class="section-divider">Classification Report</div>
      <div class="report-table-wrap">
        <table class="cr-table">
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
          <tbody>
            <tr>
              <td>Healthy (0)</td>
              <td>${cr.healthy.precision}</td>
              <td>${cr.healthy.recall}</td>
              <td>${cr.healthy['f1-score']}</td>
              <td>${sm.tn + sm.fp}</td>
            </tr>
            <tr class="cr-row-failure">
              <td><strong style="color:${C.danger}">Failure (1) ⚠</strong></td>
              <td class="${cr.failure.precision >= 0.85 ? 'cr-cell-good' : 'cr-cell-warn'}">${cr.failure.precision}</td>
              <td class="${cr.failure.recall    >= 0.75 ? 'cr-cell-good' : 'cr-cell-warn'}">${cr.failure.recall}</td>
              <td class="${cr.failure['f1-score'] >= 0.80 ? 'cr-cell-good' : 'cr-cell-warn'}"><strong>${cr.failure['f1-score']}</strong></td>
              <td>${sm.tp + sm.fn}</td>
            </tr>
            <tr>
              <td>Macro Avg</td>
              <td>${cr.macro_avg.precision}</td>
              <td>${cr.macro_avg.recall}</td>
              <td>${cr.macro_avg['f1-score']}</td>
              <td>-</td>
            </tr>
          </tbody>
        </table>
      </div>`;

    // PR curves
    const pr_traces = data.pr_curves.map(curve => ({
      type:  'scatter',
      mode:  'lines',
      name:  `${curve.name} (${curve.pr_auc})`,
      x:     curve.recall,
      y:     curve.precision,
      line:  { color: curve.color, width: curve.is_best ? 2.5 : 1.5, dash: curve.is_best ? 'solid' : 'dot' },
      hovertemplate: `<b>${curve.name}</b><br>Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>`,
    }));
    plot('ch-pr-curves', pr_traces, {
      xaxis: { title: 'Recall (Sensitivity)', range:[0,1] },
      yaxis: { title: 'Precision', range:[0,1.05] },
      margin: { l:60,r:20,t:20,b:60 },
      legend: { x:0.02, y:0.02 },
    });

    // Confusion matrix heatmap
    const cm = data.confusion_matrix;
    const ann = [];
    cm.forEach((row, i) => row.forEach((v, j) => {
      ann.push({ x: j, y: i, text: `<b>${v}</b>`, showarrow: false, font: { color: v > 100 ? C.bg : C.text, size: 18 } });
    }));
    plot('ch-conf-mat', [{
      type: 'heatmap',
      z:    cm,
      x:    ['Pred Healthy', 'Pred Failure'],
      y:    ['Actual Healthy', 'Actual Failure'],
      colorscale: [[0, C.bg],[0.5, C.surface],[1, C.accent]],
      showscale: false,
      hovertemplate: 'Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
    }], {
      annotations: ann,
      margin: { l:100,r:20,t:20,b:70 },
      xaxis: { tickfont:{color:C.sub} },
      yaxis: { tickfont:{color:C.sub} },
    });
  }

  /* ══════════════════════════════════════════════════════════
     SECTION 9 — FEATURES
  ══════════════════════════════════════════════════════════ */
  function renderFeatures(data, containerId) {
    const el = document.getElementById(containerId);
    el.innerHTML = `<div id="ch-feat-imp" style="height:420px"></div>`;

    plot('ch-feat-imp', [{
      type: 'bar',
      orientation: 'h',
      x:    data.importances,
      y:    data.features,
      text: data.importances.map(v => v.toFixed(4)),
      textposition: 'outside',
      textfont: { color: C.text, size: 10 },
      marker: {
        color: data.colors,
        line:  { color: 'rgba(0,0,0,0.3)', width: 0.5 },
      },
      hovertemplate: '<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
    }], {
      xaxis: { title: 'Relative Importance', gridcolor: 'rgba(212,212,212,0.08)' },
      title: {
        text: 'Feature Importances — <span style="color:#FF4500">■</span> Engineered  <span style="color:#525252">■</span> Raw Sensor',
        font: { color: C.sub, size: 12 },
      },
      margin: { l:160, r:60, t:50, b:50 },
      showlegend: false,
    });
  }

  /* ══════════════════════════════════════════════════════════
     SECTION 10 — ERRORS
  ══════════════════════════════════════════════════════════ */
  function renderErrors(data, containerId) {
    const el = document.getElementById(containerId);
    const sm = data.summary;

    // FN / FP probability distributions
    const fnRows = data.false_negatives;
    const fpRows = data.false_positives;
    const fnCols = fnRows.length > 0 ? Object.keys(fnRows[0]) : [];
    const fpCols = fpRows.length > 0 ? Object.keys(fpRows[0]) : [];

    el.innerHTML = `
      <div class="error-summary-grid">
        <div class="error-card"><div class="ec-val ec-rate">${sm.catch_rate}%</div><div class="ec-label">Catch Rate</div></div>
        <div class="error-card"><div class="ec-val ec-tp">${sm.true_positives}</div><div class="ec-label">True Positives</div></div>
        <div class="error-card"><div class="ec-val ec-fn">${sm.false_negatives}</div><div class="ec-label">Missed (FN)</div></div>
        <div class="error-card"><div class="ec-val ec-fp">${sm.false_positives}</div><div class="ec-label">False Alarms (FP)</div></div>
        <div class="error-card"><div class="ec-val" style="color:#D4D4D4">${sm.total_test_failures}</div><div class="ec-label">Total Test Failures</div></div>
      </div>
      <div class="chart-grid-2">
        <div class="chart-card">
          <div class="chart-card-title">Failure Modes in Missed Cases (FN)</div>
          <div class="chart-container" id="ch-fn-modes" style="height:260px"></div>
        </div>
        <div class="chart-card">
          <div class="chart-card-title">Failure Probability Distribution</div>
          <div class="chart-container" id="ch-prob-dist" style="height:260px"></div>
        </div>
      </div>
      <div class="error-table-label">❌ Missed Failures — False Negatives (top 10)</div>
      ${fnRows.length > 0 ? `<div class="et-wrap">
        <table class="et-table">
          <thead><tr>${fnCols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>
          <tbody>${fnRows.map(r=>`<tr>${fnCols.map(c=>`<td>${r[c]!==undefined?r[c]:'-'}</td>`).join('')}</tr>`).join('')}</tbody>
        </table>
      </div>` : '<p style="color:var(--text-muted);font-size:0.8rem">No false negatives on this test set.</p>'}
      <div class="error-table-label">⚠ False Alarms — False Positives (top 10)</div>
      ${fpRows.length > 0 ? `<div class="et-wrap">
        <table class="et-table">
          <thead><tr>${fpCols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>
          <tbody>${fpRows.map(r=>`<tr>${fpCols.map(c=>`<td>${r[c]!==undefined?r[c]:'-'}</td>`).join('')}</tr>`).join('')}</tbody>
        </table>
      </div>` : '<p style="color:var(--text-muted);font-size:0.8rem">No false positives on this test set.</p>'}`;

    // FN failure modes bar
    const fm = data.fn_failure_modes;
    plot('ch-fn-modes', [{
      type: 'bar',
      x: Object.keys(fm),
      y: Object.values(fm),
      text: Object.values(fm),
      textposition: 'outside',
      textfont: { color: C.text },
      marker: { color: [C.danger, C.rust, C.accent, C.healthy, C.muted] },
      hovertemplate: '<b>%{x}</b><br>%{y} missed<extra></extra>',
    }], {
      yaxis: { title: 'Count in Missed Failures' },
      margin: { l:50,r:20,t:10,b:50 },
    });

    // Probability distributions
    const probTraces = [];
    if (data.fn_probs && data.fn_probs.length > 0) {
      probTraces.push({ type:'histogram', x: data.fn_probs, name:'Missed Failures (FN)', marker:{color:C.danger, opacity:0.75}, nbinsx: 15,
        hovertemplate: 'Prob: %{x:.2f}<br>Count: %{y}<extra></extra>' });
    }
    if (data.fp_probs && data.fp_probs.length > 0) {
      probTraces.push({ type:'histogram', x: data.fp_probs, name:'False Alarms (FP)', marker:{color:C.rust, opacity:0.75}, nbinsx: 15,
        hovertemplate: 'Prob: %{x:.2f}<br>Count: %{y}<extra></extra>' });
    }
    if (probTraces.length > 0) {
      plot('ch-prob-dist', probTraces, {
        barmode: 'overlay',
        xaxis: { title: 'Model Failure Probability', range:[0,1] },
        yaxis: { title: 'Count' },
        margin: { l:50,r:20,t:10,b:50 },
      });
    }
  }

  /* ── Public API ─────────────────────────────────────────── */
  return { renderLoad, renderPreprocess, renderEDA, renderSplit, renderTrain, renderTune, renderEvaluate, renderFeatures, renderErrors };

})();
