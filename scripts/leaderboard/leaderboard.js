let appData = null;
let paretoChart = null;
let evolutionChart = null;
let refreshTimer = 5;
let activeTab = 'pareto';
let paretoZoomState = null;
let paretoHighlightWid = null;
let evolutionZoomState = null;
const codeCache = {};

function assignCandidateCode(workspaceId, codeEl) {
    if (codeCache[workspaceId] !== undefined) {
        codeEl.textContent = codeCache[workspaceId];
        return;
    }
    codeEl.textContent = 'Loading…';
    fetch('/api/candidate_code?' + new URLSearchParams({ workspace_id: workspaceId }))
        .then(function(r) {
            if (r.status === 404) {
                return r.json().then(function(j) {
                    throw new Error(j.error || 'Not found');
                });
            }
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.json();
        })
        .then(function(j) {
            var t = (j.code !== undefined && j.code !== null) ? j.code : '';
            codeCache[workspaceId] = t;
            codeEl.textContent = t || '(no code)';
        })
        .catch(function(err) {
            codeEl.textContent = 'Failed to load code: ' + err.message;
        });
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
function formatMetric(m) {
    if (m === null || m === undefined) return 'N/A';
    if (m === 'Infinity' || m === Infinity) return '\u221e';
    if (m === '-Infinity' || m === -Infinity) return '-\u221e';
    return parseFloat(m).toFixed(4);
}
function parseVerdict(summary) {
    if (!summary) return { verdict: null, body: '' };
    const m = summary.match(/^Followed assigned approach:\s*(yes|partial|no)\.\s*/i);
    if (!m) return { verdict: null, body: summary };
    return { verdict: m[1].toLowerCase(), body: summary.slice(m[0].length) };
}
function verdictBadge(verdict) {
    if (verdict === 'yes')     return '<span class="badge bg-success">followed: yes</span>';
    if (verdict === 'partial') return '<span class="badge bg-warning text-dark">followed: partial</span>';
    if (verdict === 'no')      return '<span class="badge bg-danger">followed: no</span>';
    return '<span class="badge bg-secondary">verdict: n/a</span>';
}
function summaryBlock(summary) {
    const { verdict, body } = parseVerdict(summary);
    return '<div class="mb-2"><strong>Summary:</strong> ' + verdictBadge(verdict) +
        '<div class="mt-1 small">' + escapeHtml(body || '(no summary)') + '</div></div>';
}

function formatTimestamp(ts) {
    if (!ts || ts === 'N/A') return 'N/A';
    const d = new Date(ts);
    if (isNaN(d.getTime())) return ts;
    const pad = n => String(n).padStart(2, '0');
    return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate())
        + ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
}

function parseMetric(m) {
    if (m === "Infinity" || m === Infinity) return Infinity;
    if (m === "-Infinity" || m === -Infinity) return -Infinity;
    if (m === null || m === undefined) return Infinity;
    return parseFloat(m);
}

// Tab switching
document.querySelectorAll('[data-tab]').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.nav-link').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.style.display = 'none');
        this.classList.add('active');
        activeTab = this.getAttribute('data-tab');
        document.getElementById('tab-' + activeTab).style.display = 'block';
    });
});

// Candidate detail view
function showCandidate(workspaceId) {
    if (!appData) return;
    const cand = appData.candidates[workspaceId];
    if (!cand) return;
    document.getElementById('selectedCandidate').textContent = workspaceId;
    let hpHtml = '';
    if (cand.hp_results && cand.hp_results.length) {
        hpHtml = '<table class="table table-sm mt-2"><thead><tr><th>HP#</th><th>Metric</th><th>Complexity</th></tr></thead><tbody>';
        cand.hp_results.forEach(hp => {
            hpHtml += '<tr><td>' + hp.hp_index + '</td><td>' + formatMetric(hp.metric) + '</td><td>' + formatMetric(hp.complexity) + '</td></tr>';
        });
        hpHtml += '</tbody></table>';
    }
    const ideaDescIdeas = (appData.cluster_descriptions || {})[cand.cluster] || '(no description)';
    document.getElementById('candidateInfo').innerHTML =
        '<div class="row"><div class="col-md-3"><strong>Status:</strong> ' +
        (cand.success ? '<span class="badge bg-success">Success</span>' : '<span class="badge bg-danger">Failed</span>') +
        '</div><div class="col-md-3"><strong>Gen:</strong> ' + (cand.generation || 0) +
        '</div><div class="col-md-3"><strong>Idea:</strong> <span class="cluster-name">' + escapeHtml(cand.cluster || '') + '</span></div>' +
        '<div class="col-md-3"><strong>HP configs:</strong> ' + (cand.hp_results ? cand.hp_results.length : 0) + '</div></div>' +
        '<div class="mt-2 mb-2"><strong>Idea description:</strong> ' + escapeHtml(ideaDescIdeas) + '</div>' +
        summaryBlock(cand.summary) +
        hpHtml;
    const codeEl = document.getElementById('candidateCode');
    codeEl.style.display = 'block';
    assignCandidateCode(workspaceId, codeEl);
}

// Render stats
function renderStats() {
    const d = appData;
    document.getElementById('statTotal').textContent = d.total_candidates;
    document.getElementById('statSuccess').textContent = d.successful_candidates;
    document.getElementById('statFailed').textContent = d.total_candidates - d.successful_candidates;
    document.getElementById('statClusters').textContent = d.num_clusters;
    document.getElementById('statGens').textContent = d.generation_stats.length;
    document.getElementById('statBest').textContent = d.best_metric !== null ? formatMetric(d.best_metric) : 'N/A';
    document.getElementById('lastUpdated').textContent = formatTimestamp(d.last_updated);
}

// Cluster colors
const clusterColors = [
    'rgba(13,110,253,0.7)', 'rgba(40,167,69,0.7)', 'rgba(220,53,69,0.7)',
    'rgba(255,193,7,0.7)', 'rgba(111,66,193,0.7)', 'rgba(23,162,184,0.7)',
    'rgba(253,126,20,0.7)', 'rgba(108,117,125,0.7)', 'rgba(0,123,255,0.7)',
    'rgba(102,16,242,0.7)',
];

// Pareto zoom reset
let suppressZoomSave = false;
function resetParetoZoom() {
    paretoZoomState = null;
    if (paretoChart) {
        suppressZoomSave = true;
        delete paretoChart.options.scales.x.min;
        delete paretoChart.options.scales.x.max;
        delete paretoChart.options.scales.y.min;
        delete paretoChart.options.scales.y.max;
        paretoChart.resetZoom('none');
        paretoChart.update('none');
        suppressZoomSave = false;
        document.getElementById('resetZoomBtn').style.display = 'none';
    }
}

// Evolution chart zoom reset
function resetEvolutionZoom() {
    evolutionZoomState = null;
    if (evolutionChart) {
        suppressZoomSave = true;
        delete evolutionChart.options.scales.x.min;
        delete evolutionChart.options.scales.x.max;
        delete evolutionChart.options.scales.y.min;
        delete evolutionChart.options.scales.y.max;
        delete evolutionChart.options.scales.y1.min;
        delete evolutionChart.options.scales.y1.max;
        evolutionChart.resetZoom('none');
        evolutionChart.update('none');
        suppressZoomSave = false;
        document.getElementById('resetEvolutionZoomBtn').style.display = 'none';
    }
}

// Pareto point highlighting
function highlightParetoPoints(wid) {
    paretoHighlightWid = wid;
    if (!paretoChart) return;
    paretoChart.data.datasets.forEach(ds => {
        if (!ds.data.length || !ds.data[0].workspace_id) return;
        const radii = [];
        const bgColors = [];
        const borderColors = [];
        ds.data.forEach(p => {
            if (p.workspace_id === wid) {
                radii.push(9);
                bgColors.push(ds.backgroundColor.replace('0.7', '1'));
                borderColors.push('rgba(0,0,0,0.9)');
            } else {
                radii.push(4);
                bgColors.push(ds.backgroundColor.replace('0.7', '0.2'));
                borderColors.push(ds.borderColor.replace('1', '0.2'));
            }
        });
        ds.pointRadius = radii;
        ds.pointBackgroundColor = bgColors;
        ds.pointBorderColor = borderColors;
        ds.pointBorderWidth = ds.data.map(p => p.workspace_id === wid ? 2 : 1);
    });
    paretoChart.update();
}
function clearParetoHighlight() {
    paretoHighlightWid = null;
    if (!paretoChart) return;
    paretoChart.data.datasets.forEach(ds => {
        if (!ds.data.length || !ds.data[0].workspace_id) return;
        ds.pointRadius = 5;
        ds.pointBackgroundColor = ds.backgroundColor;
        ds.pointBorderColor = ds.borderColor;
        ds.pointBorderWidth = 1;
    });
    paretoChart.update();
}

// Pareto detail panel
function showParetoDetail(workspaceId, hpIndex) {
    if (!appData) return;
    const cand = appData.candidates[workspaceId];
    if (!cand) return;
    const descriptions = appData.cluster_descriptions || {};
    const ideaDesc = descriptions[cand.cluster] || '(no description)';

    // Locate the specific HP result for this scatter point.
    const hpResults = cand.hp_results || [];
    let hp = null;
    if (typeof hpIndex === 'number') {
        hp = hpResults.find(h => h.hp_index === hpIndex) || null;
    }
    if (!hp && hpResults.length) hp = hpResults[0];
    const params = (hp && hp.params) || {};

    const badgeText = (typeof hpIndex === 'number')
        ? (workspaceId + ' · hp=' + hpIndex)
        : workspaceId;
    document.getElementById('paretoDetailBadge').textContent = badgeText;

    const metricStr = hp ? Number(hp.metric).toPrecision(6) : '—';
    const complexityStr = hp ? Number(hp.complexity).toPrecision(6) : '—';
    document.getElementById('paretoDetailInfo').innerHTML =
        '<div class="row mb-2">' +
        '<div class="col-md-2"><strong>Status:</strong> ' +
        (cand.success ? '<span class="badge bg-success">Success</span>' : '<span class="badge bg-danger">Failed</span>') +
        '</div><div class="col-md-2"><strong>Gen:</strong> ' + (cand.generation || 0) +
        '</div><div class="col-md-2"><strong>Idea:</strong> <span class="cluster-name">' + escapeHtml(cand.cluster || '') + '</span>' +
        '</div><div class="col-md-3"><strong>Metric:</strong> ' + escapeHtml(metricStr) +
        '</div><div class="col-md-3"><strong>Complexity:</strong> ' + escapeHtml(complexityStr) +
        '</div></div>' +
        '<div class="mb-2"><strong>Idea description:</strong> ' + escapeHtml(ideaDesc) + '</div>' +
        summaryBlock(cand.summary);

    // Render hyperparameters for the selected point.
    const paramsEl = document.getElementById('paretoDetailParams');
    const paramKeys = Object.keys(params);
    if (!paramKeys.length) {
        paramsEl.innerHTML = '<div class="text-muted small">No tuned hyperparameters '
            + '(baseline defaults from <code>solution.py</code>).</div>';
    } else {
        let html = '<table class="table table-sm table-striped mb-0">'
            + '<thead><tr><th>Name</th><th>Value</th></tr></thead><tbody>';
        paramKeys.sort().forEach(k => {
            const v = params[k];
            const vStr = (typeof v === 'number') ? String(v) : JSON.stringify(v);
            html += '<tr><td><code>' + escapeHtml(k) + '</code></td>'
                + '<td><code>' + escapeHtml(vStr) + '</code></td></tr>';
        });
        html += '</tbody></table>';
        paramsEl.innerHTML = html;
    }

    const codeEl = document.getElementById('paretoDetailCode');
    codeEl.style.display = 'block';
    assignCandidateCode(workspaceId, codeEl);
    document.getElementById('paretoDetailRow').style.display = 'block';
}
function closeParetoDetail() {
    document.getElementById('paretoDetailRow').style.display = 'none';
    clearParetoHighlight();
}

// Pareto chart
function renderParetoChart() {
    if (paretoChart) {
        const xScale = paretoChart.scales.x;
        const yScale = paretoChart.scales.y;
        if (paretoZoomState) {
            paretoZoomState = { xMin: xScale.min, xMax: xScale.max, yMin: yScale.min, yMax: yScale.max };
        }
        paretoChart.destroy();
        paretoChart = null;
    }
    const sp = appData.scatter_points || [];
    const pp = appData.pareto_points || [];
    if (!sp.length) return;

    const clusterSet = [...new Set(sp.map(p => p.cluster))].sort((a,b) => {
        const na = parseInt(a), nb = parseInt(b);
        if (!isNaN(na) && !isNaN(nb)) return na - nb;
        return String(a).localeCompare(String(b));
    });
    const colorMap = {};
    clusterSet.forEach((c, i) => { colorMap[c] = clusterColors[i % clusterColors.length]; });

    const datasets = clusterSet.map(cl => ({
        label: 'Idea ' + cl,
        data: sp.filter(p => p.cluster === cl).map(p => ({
            x: p.x, y: p.y,
            workspace_id: p.workspace_id,
            hp_index: p.hp_index,
            cluster: p.cluster,
        })),
        backgroundColor: colorMap[cl],
        borderColor: colorMap[cl].replace('0.7', '1'),
        pointRadius: 5,
        pointHoverRadius: 8,
    }));

    // Sort Pareto front for step line
    const sortedPareto = [...pp].sort((a, b) => a.x - b.x);
    datasets.push({
        label: 'Pareto Front',
        data: sortedPareto.map(p => ({ x: p.x, y: p.y })),
        type: 'line',
        borderColor: 'rgba(0,0,0,0.8)',
        backgroundColor: 'rgba(0,0,0,0.05)',
        borderWidth: 2,
        borderDash: [6, 3],
        pointRadius: 0,
        fill: false,
        stepped: 'after',
        order: -1,
    });

    const hib = appData.higher_is_better;
    const ctx = document.getElementById('paretoChart').getContext('2d');
    paretoChart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            onClick: function(event, elements) {
                if (!elements.length) {
                    clearParetoHighlight();
                    return;
                }
                const el = elements[0];
                const point = paretoChart.data.datasets[el.datasetIndex].data[el.index];
                if (point && point.workspace_id) {
                    showParetoDetail(point.workspace_id, point.hp_index);
                    highlightParetoPoints(point.workspace_id);
                }
            },
            scales: {
                x: Object.assign(
                    { title: { display: true, text: 'Metric (' + (hib ? 'higher is better' : 'lower is better') + ')' } },
                    paretoZoomState ? { min: paretoZoomState.xMin, max: paretoZoomState.xMax } : {}
                ),
                y: Object.assign(
                    { title: { display: true, text: 'Complexity (lower is better)' } },
                    paretoZoomState ? { min: paretoZoomState.yMin, max: paretoZoomState.yMax } : {}
                ),
            },
            plugins: {
                title: { display: true, text: 'Solution Space — Metric vs Complexity' },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            const p = ctx.raw;
                            if (p.workspace_id) {
                                return p.workspace_id + ':' + p.hp_index +
                                    ' — Metric: ' + parseFloat(p.x).toFixed(4) +
                                    ', Complexity: ' + parseFloat(p.y).toFixed(4);
                            }
                            return 'Metric: ' + parseFloat(p.x).toFixed(4) +
                                ', Complexity: ' + parseFloat(p.y).toFixed(4);
                        }
                    }
                },
                legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } },
                zoom: {
                    zoom: {
                        drag: {
                            enabled: true,
                            backgroundColor: 'rgba(13,110,253,0.15)',
                            borderColor: 'rgba(13,110,253,0.6)',
                            borderWidth: 1,
                            threshold: 5,
                        },
                        mode: 'xy',
                        onZoomComplete: function(context) {
                            if (suppressZoomSave) return;
                            const chart = context.chart;
                            paretoZoomState = {
                                xMin: chart.scales.x.min,
                                xMax: chart.scales.x.max,
                                yMin: chart.scales.y.min,
                                yMax: chart.scales.y.max,
                            };
                            document.getElementById('resetZoomBtn').style.display = 'inline-block';
                        },
                    },
                },
            },
        },
    });
    if (paretoZoomState) {
        document.getElementById('resetZoomBtn').style.display = 'inline-block';
    }
    if (paretoHighlightWid) {
        highlightParetoPoints(paretoHighlightWid);
    }
}

// Ideas/clusters tab — grouped by generation
function renderClusters() {
    // Group candidates by cluster
    const clusters = {};
    for (const [wid, c] of Object.entries(appData.candidates)) {
        const cl = c.cluster;
        const key = (cl === undefined || cl === null) ? 'default' : cl;
        if (!clusters[key]) clusters[key] = [];
        clusters[key].push(c);
    }

    // Determine generation per cluster (= min generation of its candidates).
    // Clusters with no candidates fall back to 0.
    const clusterGen = {};
    for (const [k, cands] of Object.entries(clusters)) {
        let g = null;
        for (const c of cands) {
            const cg = c.generation || 0;
            if (g === null || cg < g) g = cg;
        }
        clusterGen[k] = g === null ? 0 : g;
    }

    // Group cluster keys by generation
    const byGen = {};
    for (const k of Object.keys(clusters)) {
        const g = clusterGen[k];
        if (!byGen[g]) byGen[g] = [];
        byGen[g].push(k);
    }
    const sortClusterKeys = (arr) => arr.sort((a, b) => {
        const na = parseInt(a), nb = parseInt(b);
        if (!isNaN(na) && !isNaN(nb)) return na - nb;
        return String(a).localeCompare(String(b));
    });
    for (const g of Object.keys(byGen)) sortClusterKeys(byGen[g]);

    const generations = Object.keys(byGen).map(Number).sort((a, b) => a - b);
    document.getElementById('clusterCount').textContent =
        Object.keys(clusters).length;

    const descriptions = appData.cluster_descriptions || {};
    let html = '';
    for (const gen of generations) {
        const clusterKeys = byGen[gen];
        const totalCands = clusterKeys.reduce(
            (s, k) => s + clusters[k].length, 0
        );
        const genId = 'gen-section-' + gen;
        html += '<div class="generation-header" onclick="toggleCluster(\'' + genId + '\')">' +
            '<span>Generation ' + gen + ' &mdash; ' + clusterKeys.length +
            ' idea' + (clusterKeys.length === 1 ? '' : 's') + '</span>' +
            '<span class="badge bg-light text-dark">' + totalCands + ' candidate' +
            (totalCands === 1 ? '' : 's') + '</span></div>' +
            '<div id="' + genId + '" class="collapse show">';

        for (const cn of clusterKeys) {
            const cands = clusters[cn];
            const desc = descriptions[cn];
            const descText = (desc != null && desc !== '') ? desc : '(no description)';
            let items = '';
            for (const c of cands) {
                const statusClass = c.success ? 'candidate-success' : 'candidate-failed';
                let bestM = 'N/A';
                if (c.success && c.hp_results && c.hp_results.length) {
                    const metrics = c.hp_results.map(h => h.metric);
                    bestM = formatMetric(appData.higher_is_better ? Math.max(...metrics) : Math.min(...metrics));
                }
                const verdictPill = (function() {
                    const v = parseVerdict(c.summary).verdict;
                    if (v === 'yes')     return '<span class="badge bg-success ms-2" title="Followed assigned approach: yes">y</span>';
                    if (v === 'partial') return '<span class="badge bg-warning text-dark ms-2" title="Followed assigned approach: partial">~</span>';
                    if (v === 'no')      return '<span class="badge bg-danger ms-2" title="Followed assigned approach: no">n</span>';
                    return '';
                })();
                items += '<div class="candidate-item ' + statusClass + '" onclick="showCandidate(\''+c.workspace_id+'\')">' +
                    '<div class="d-flex justify-content-between align-items-center">' +
                    '<span><small class="text-muted">Gen ' + (c.generation || 0) + '</small> ' + escapeHtml(c.workspace_id) + '</span>' +
                    '<span><span class="metric-value">' + bestM + '</span>' + verdictPill + '</span></div></div>';
            }
            html += '<div class="cluster-section mb-3">' +
                '<div class="cluster-header">' +
                '<div class="d-flex justify-content-between align-items-start">' +
                '<strong class="cluster-name">Idea ' + escapeHtml(String(cn)) + '</strong>' +
                '<span class="badge bg-secondary">' + cands.length + '</span></div>' +
                '<div class="cluster-desc">' + escapeHtml(descText) + '</div></div>' +
                items + '</div>';
        }
        html += '</div>';
    }
    document.getElementById('clustersContainer').innerHTML = html;
}
function toggleCluster(id) {
    document.getElementById(id).classList.toggle('show');
}

// Generation stats
function renderGenStats() {
    let html = '';
    for (const g of appData.generation_stats) {
        html += '<tr><td><strong>' + g.generation + '</strong></td>' +
            '<td>' + g.total + '</td><td class="text-success">' + g.successful + '</td>' +
            '<td class="text-danger">' + g.failed + '</td></tr>';
    }
    document.getElementById('genStatsBody').innerHTML = html;
}

// Evolution chart
function renderEvolutionChart() {
    if (evolutionChart) {
        const xScale = evolutionChart.scales.x;
        const yScale = evolutionChart.scales.y;
        const y1Scale = evolutionChart.scales.y1;
        if (evolutionZoomState) {
            evolutionZoomState = {
                xMin: xScale.min, xMax: xScale.max,
                yMin: yScale.min, yMax: yScale.max,
                y1Min: y1Scale.min, y1Max: y1Scale.max,
            };
        }
        evolutionChart.destroy();
        evolutionChart = null;
    }
    const gs = appData.generation_stats;
    if (!gs.length) return;
    const hib = appData.higher_is_better;
    const complexityLabel = 'Complexity at Best Metric So Far';
    const ctx = document.getElementById('evolutionChart').getContext('2d');
    evolutionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: gs.map(g => 'Gen ' + g.generation),
            datasets: [
                {
                    label: 'Best Metric So Far',
                    data: gs.map(g => g.best_metric),
                    borderColor: 'rgba(13,110,253,1)',
                    backgroundColor: 'rgba(13,110,253,0.1)',
                    fill: true,
                    yAxisID: 'y',
                    tension: 0.2,
                    pointRadius: 5,
                    pointHoverRadius: 8,
                    borderWidth: 2,
                },
                {
                    label: complexityLabel,
                    data: gs.map(g => g.best_complexity),
                    borderColor: 'rgba(40,167,69,1)',
                    backgroundColor: 'rgba(40,167,69,0.1)',
                    fill: true,
                    yAxisID: 'y1',
                    tension: 0.2,
                    pointRadius: 5,
                    pointHoverRadius: 8,
                    borderWidth: 2,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: Object.assign(
                    {},
                    evolutionZoomState ? { min: evolutionZoomState.xMin, max: evolutionZoomState.xMax } : {}
                ),
                y: Object.assign(
                    {
                        type: 'linear',
                        position: 'left',
                        title: { display: true, text: 'Best Metric So Far (' + (hib ? 'higher is better' : 'lower is better') + ')', color: 'rgba(13,110,253,1)' },
                        ticks: { color: 'rgba(13,110,253,0.8)' },
                    },
                    evolutionZoomState ? { min: evolutionZoomState.yMin, max: evolutionZoomState.yMax } : {}
                ),
                y1: Object.assign(
                    {
                        type: 'linear',
                        position: 'right',
                        title: { display: true, text: complexityLabel, color: 'rgba(40,167,69,1)' },
                        ticks: { color: 'rgba(40,167,69,0.8)' },
                        grid: { drawOnChartArea: false },
                    },
                    evolutionZoomState ? { min: evolutionZoomState.y1Min, max: evolutionZoomState.y1Max } : {}
                ),
            },
            plugins: {
                title: { display: true, text: 'Evolution Over Generations' },
                zoom: {
                    zoom: {
                        drag: {
                            enabled: true,
                            backgroundColor: 'rgba(13,110,253,0.15)',
                            borderColor: 'rgba(13,110,253,0.6)',
                            borderWidth: 1,
                            threshold: 5,
                        },
                        mode: 'xy',
                        onZoomComplete: function(context) {
                            if (suppressZoomSave) return;
                            const chart = context.chart;
                            evolutionZoomState = {
                                xMin: chart.scales.x.min,
                                xMax: chart.scales.x.max,
                                yMin: chart.scales.y.min,
                                yMax: chart.scales.y.max,
                                y1Min: chart.scales.y1.min,
                                y1Max: chart.scales.y1.max,
                            };
                            document.getElementById('resetEvolutionZoomBtn').style.display = 'inline-block';
                        },
                    },
                },
            },
        },
    });
    if (evolutionZoomState) {
        document.getElementById('resetEvolutionZoomBtn').style.display = 'inline-block';
    }
}

function renderQuery() {
    document.getElementById('queryBox').textContent = appData.query || '';
}


// ============ Agent Journals ============
let journalAgents = [];
let currentJournalId = null;

function fetchJournalAgents() {
    fetch('/api/journals')
        .then(r => r.json())
        .then(agents => {
            journalAgents = agents;
            populateJournalGenSelect();
        })
        .catch(err => console.error('Journal agents fetch error:', err));
}

function populateJournalGenSelect() {
    const genSelect = document.getElementById('journalGenSelect');
    const prevGen = genSelect.value;
    const gens = [...new Set(journalAgents.map(a => a.generation))].sort((a, b) => a - b);
    let html = '';
    gens.forEach(g => { html += `<option value="${g}">Gen ${g}</option>`; });
    genSelect.innerHTML = html;
    if (prevGen !== '' && gens.includes(parseInt(prevGen))) {
        genSelect.value = prevGen;
    } else if (gens.length > 0) {
        genSelect.value = gens[0];
    }
    onJournalGenChange();
}

function onJournalGenChange() {
    const gen = document.getElementById('journalGenSelect').value;
    const agentSelect = document.getElementById('journalAgentSelect');
    const prevAgent = agentSelect.value;
    if (gen === '') {
        agentSelect.innerHTML = '';
        return;
    }
    const filtered = journalAgents.filter(a => a.generation === parseInt(gen));
    let html = '';
    filtered.forEach(a => { html += `<option value="${a.id}">${a.id}</option>`; });
    agentSelect.innerHTML = html;
    if (prevAgent && filtered.some(a => a.id === prevAgent)) {
        agentSelect.value = prevAgent;
    } else if (filtered.length > 0) {
        agentSelect.value = filtered[0].id;
    }
    onJournalAgentChange();
}

function onJournalAgentChange() {
    const agentId = document.getElementById('journalAgentSelect').value;
    if (!agentId) {
        currentJournalId = null;
        lastJournalContent = null;
        document.getElementById('journalContent').innerHTML = '<span class="text-muted">Select a generation and agent to view the journal log.</span>';
        return;
    }
    if (agentId === currentJournalId) return;
    currentJournalId = agentId;
    lastJournalContent = null;
    refreshJournal();
}

let lastJournalContent = null;

const journalTagColors = {
    'QUERY':         '#61afef',
    'AI':            '#c678dd',
    'TOOL_CALL':     '#e5c07b',
    'TOOL_RESPONSE': '#98c379',
    'POST_EVAL':     '#56b6c2',
    'POST_HP_TUNE':  '#56b6c2',
    'TIMEOUT':       '#e06c75',
};
const journalTimestampColor = '#5c6370';
const journalTagRe = /^(\[[^\]]+\]) (\[[A-Z_]+\])/;

function colorizeJournal(raw) {
    const lines = raw.split('\n');
    let out = '';
    for (let i = 0; i < lines.length; i++) {
        const m = lines[i].match(journalTagRe);
        if (m) {
            const ts = escapeHtml(m[1]);
            const tag = m[2];
            const tagName = tag.slice(1, -1);
            const tagColor = journalTagColors[tagName] || '#abb2bf';
            const rest = escapeHtml(lines[i].slice(m[0].length));
            out += `<span style="color:${journalTimestampColor}">${ts}</span> <span style="color:${tagColor};font-weight:600">${escapeHtml(tag)}</span>${rest}`;
        } else {
            out += escapeHtml(lines[i]);
        }
        if (i < lines.length - 1) out += '\n';
    }
    return out;
}

function refreshJournal() {
    if (!currentJournalId) return;
    fetch('/api/journal?id=' + encodeURIComponent(currentJournalId))
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                document.getElementById('journalContent').textContent = 'Error: ' + data.error;
                lastJournalContent = null;
                return;
            }
            const content = data.content || '(empty log)';
            if (content === lastJournalContent) return;
            const el = document.getElementById('journalContent');
            el.innerHTML = colorizeJournal(content);
            if (document.getElementById('journalAutoScroll').checked) {
                el.scrollTop = el.scrollHeight;
            }
            lastJournalContent = content;
        })
        .catch(err => {
            document.getElementById('journalContent').textContent = 'Fetch error: ' + err;
            lastJournalContent = null;
        });
}

// Data fetching
function fetchData() {
    fetch('/api/data')
        .then(r => r.json())
        .then(data => {
            appData = data;
            renderStats();
            renderParetoChart();
            renderClusters();
            renderGenStats();
            renderEvolutionChart();
            renderQuery();
            document.querySelector('.refresh-indicator').classList.remove('error');
        })
        .catch(err => {
            console.error('Fetch error:', err);
            document.querySelector('.refresh-indicator').classList.add('error');
        });
}

setInterval(() => {
    refreshTimer--;
    if (refreshTimer <= 0) {
        refreshTimer = 5;
        fetchData();
        fetchJournalAgents();
        if (activeTab === 'journals' && currentJournalId) refreshJournal();
    }
    document.getElementById('refreshCountdown').textContent = refreshTimer;
}, 1000);

fetchData();
fetchJournalAgents();
