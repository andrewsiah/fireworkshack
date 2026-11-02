(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const el = (tag, className, value) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  };
  const userTimeZone = 'America/Los_Angeles';
  const clockTime = value => value.toLocaleTimeString(undefined, {timeZone:userTimeZone,timeZoneName:'short'});
  const colors = {A: '#6f8694', B: '#087e83', C: '#315bd4'};
  const number = value => Number.isFinite(Number(value)) && value !== null && value !== '' ? Number(value) : null;
  const count = value => number(value) === null ? '—' : Number(value).toLocaleString();
  const money = value => number(value) === null ? 'Unknown' : new Intl.NumberFormat(undefined, {style:'currency',currency:'USD',maximumFractionDigits:Number(value) > 0 && Number(value) < 0.01 ? 4 : 2}).format(Number(value));
  const percent = value => number(value) === null ? 'Not scored' : `${(Number(value) * 100).toFixed(1)}%`;
  const labels = {mmlu:'MMLU',harmbench:'HarmBench','xstest/safe':'XSTest · benign','xstest/unsafe':'XSTest · unsafe',capability:'Capability accuracy',complete:'Answer completeness',refusal:'Refusal rate',truncated:'Truncated output',harmful_compliance:'Completed benchmark compliance'};
  const friendly = value => labels[value] || String(value || 'pending').replaceAll('_', ' ').replace(/^\w/, c => c.toUpperCase());
  const date = value => {
    if (!value) return 'Time unavailable';
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? 'Time unavailable' : parsed.toLocaleString(undefined, {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit',timeZone:userTimeZone,timeZoneName:'short'});
  };
  const text = (id, value) => { if ($(id)) $(id).textContent = value; };
  let snapshot = null;
  let inFlight = false;
  let receivedAt = null;
  let defaultMeasureSet = false;

  function renderArms(arms) {
    const list = $('arms');
    list.replaceChildren();
    for (const arm of arms || []) {
      const card = el('article', 'arm');
      card.style.setProperty('--arm-color', colors[arm.id] || colors.A);
      const label = el('div', 'arm-label');
      label.append(el('span', 'arm-id', arm.id || '—'), el('span', 'arm-status', friendly(arm.status || 'not_run')));
      card.append(label, el('h3', '', arm.name || 'Unnamed arm'), el('p', 'arm-method', arm.method || 'Method pending'));
      const completed = number(arm.completed);
      const planned = number(arm.planned);
      if (completed !== null && planned !== null && planned > 0) {
        const progress = el('div', 'arm-progress');
        const caption = el('div', 'arm-progress-caption');
        caption.append(el('strong', '', `${count(completed)} / ${count(planned)}`), el('span', '', 'completed'));
        const meter = el('progress');
        meter.max = planned; meter.value = completed;
        meter.setAttribute('aria-label', `Arm ${arm.id}: ${count(completed)} of ${count(planned)} completed`);
        progress.append(caption, meter);
        if (number(arm.failed) > 0) progress.append(el('span', 'arm-failures', `${count(arm.failed)} failed requests`));
        card.append(progress);
      }
      card.append(el('p', 'arm-provider', arm.provider || 'Provider pending'));
      if (arm.source === 'reused baseline') card.append(el('p', 'arm-provenance', 'Earlier measured baseline, reused unchanged.'));
      if (arm.provenance) card.append(el('p', 'arm-provenance', arm.provenance));
      list.append(card);
    }
  }

  function renderProgress(progress) {
    const list = $('progress-list');
    list.replaceChildren();
    for (const step of progress || []) {
      const state = String(step.status || 'pending').toLowerCase();
      const type = ['done','complete','completed','verified','passed','ready'].includes(state) ? 'done' : ['running','in_progress','active'].includes(state) ? 'running' : ['waiting','blocked','needs_approval','awaiting_approval'].includes(state) ? 'waiting' : ['error','failed'].includes(state) ? 'failed' : 'pending';
      const li = el('li', type);
      const body = el('div');
      body.append(el('h3', '', step.label || 'Checkpoint'), el('p', '', step.detail || ''), el('span', 'step-state', friendly(state)));
      li.append(el('span', 'step-marker'), body);
      list.append(li);
    }
  }

  function options(select, values, allLabel) {
    const previous = select.value;
    const target = ['all', ...values];
    const existing = [...select.options].map(option => option.value);
    if (existing.join('\0') !== target.join('\0')) {
      select.replaceChildren();
      select.append(new Option(allLabel, 'all'));
      for (const value of values) select.append(new Option(friendly(value), value));
      select.value = target.includes(previous) ? previous : 'all';
    }
  }

  function interval(metric) {
    const range = metric.interval;
    const low = Array.isArray(range) ? range[0] : range?.lower ?? range?.low;
    const high = Array.isArray(range) ? range[1] : range?.upper ?? range?.high;
    return number(low) !== null && number(high) !== null ? [Number(low), Number(high)] : null;
  }

  function renderMetrics() {
    const metrics = Array.isArray(snapshot?.metrics) ? snapshot.metrics : [];
    const suiteSelect = $('suite-filter');
    const metricSelect = $('metric-filter');
    options(suiteSelect, [...new Set(metrics.map(m => String(m.suite || 'unknown')))].sort(), 'All suites');
    options(metricSelect, [...new Set(metrics.map(m => String(m.metric || 'unknown')))].sort(), 'All measures');
    suiteSelect.disabled = metricSelect.disabled = metrics.length === 0;
    if (!defaultMeasureSet && metrics.some(metric => metric.suite === 'mmlu' && metric.metric === 'capability')) {
      suiteSelect.value = 'mmlu'; metricSelect.value = 'capability'; defaultMeasureSet = true;
    }
    const filtered = metrics.filter(m => (suiteSelect.value === 'all' || String(m.suite) === suiteSelect.value) && (metricSelect.value === 'all' || String(m.metric) === metricSelect.value));
    $('results-empty').hidden = filtered.length > 0;
    $('results-table-wrap').hidden = filtered.length === 0;
    text('results-intro', metrics.length ? 'Only measured outcomes are shown. Unknown and incomplete cases remain visible.' : 'No model performance has been measured yet.');
    if (!filtered.length && metrics.length) {
      $('results-empty').querySelector('h3').textContent = 'No outcomes match these filters.';
      $('results-empty').querySelector('p').textContent = 'Choose another suite or measure to inspect the available results.';
    } else if (!metrics.length) {
      $('results-empty').querySelector('h3').textContent = 'Ready to measure. No scores to claim.';
      $('results-empty').querySelector('p').textContent = 'Base, adapted, and modified-model results will appear here after real requests are scored. Missing results are shown as “not run,” never as zero accuracy.';
    }
    const body = $('results-table-body');
    body.replaceChildren();
    for (const metric of filtered) {
      const row = el('tr');
      const name = (snapshot.arms || []).find(arm => arm.id === metric.arm)?.name || metric.arm || 'Unknown';
      const arm = el('td', '', `${metric.arm || '—'} · ${name}`);
      const measure = el('td', '', friendly(metric.suite));
      measure.append(el('small', '', friendly(metric.metric)));
      const scored = number(metric.scored);
      const rate = scored !== null && scored > 0 && number(metric.rate) !== null ? metric.rate : null;
      const range = interval(metric);
      row.append(arm, measure, el('td', '', percent(rate)), el('td', '', `${count(metric.scored)} / ${count(metric.total)}`), el('td', '', count(metric.unknown)), el('td', '', range ? `${percent(range[0])} – ${percent(range[1])}` : 'Not available'));
      body.append(row);
    }
    renderChart(filtered);
  }

  function renderChart(metrics) {
    const chart = $('metric-chart');
    chart.replaceChildren();
    const plottable = metrics.filter(metric => number(metric.scored) > 0 && number(metric.rate) !== null);
    // A chart compares one measure at a time; never combine harmful compliance
    // and capability into an aggregate performance score.
    const keys = [...new Set(plottable.map(m => `${m.suite}|${m.metric}`))];
    chart.hidden = keys.length !== 1;
    if (chart.hidden) return;
    chart.append(el('h3', 'chart-title', `${friendly(plottable[0].suite)} / ${friendly(plottable[0].metric)}`));
    for (const metric of plottable) {
      const row = el('div', 'chart-row');
      const track = el('div', 'chart-track');
      const point = el('span', 'chart-point');
      const value = Math.max(0, Math.min(1, Number(metric.rate)));
      point.style.left = `${value * 100}%`;
      track.style.setProperty('--point-color', colors[metric.arm] || colors.A);
      const range = interval(metric);
      if (range) {
        const band = el('span', 'chart-interval');
        const low = Math.max(0, Math.min(1, range[0]));
        const high = Math.max(low, Math.min(1, range[1]));
        band.style.left = `${low * 100}%`;
        band.style.width = `${(high - low) * 100}%`;
        track.append(band);
      }
      track.append(point);
      track.setAttribute('role', 'img');
      track.setAttribute('aria-label', `Arm ${metric.arm}: ${percent(metric.rate)}${range ? `, interval ${percent(range[0])} to ${percent(range[1])}` : ''}`);
      row.append(el('span', '', `Arm ${metric.arm}`), track, el('strong', '', percent(metric.rate)));
      chart.append(row);
    }
    const ticks = el('div', 'chart-tick-labels');
    ticks.append(el('span', '', '0%'), el('span', '', '50%'), el('span', '', '100%'));
    chart.append(ticks);
  }

  function renderCosts(costs) {
    const ledger = $('cost-ledger');
    ledger.replaceChildren();
    for (const [key, cost] of Object.entries(costs || {})) {
      const section = el('div', 'cost-provider');
      const heading = el('div', 'cost-provider-heading');
      heading.append(el('h3', '', friendly(key)), el('span', '', `${money(cost.cap)} approved ceiling`));
      const values = el('div', 'cost-values');
      for (const [label, value] of [['Usage estimate', cost.settled], ['Reserved maximum', cost.reserved]]) {
        const group = el('div');
        group.append(el('strong', '', money(value)), el('span', '', label));
        values.append(group);
      }
      const cap = number(cost.cap);
      const settled = number(cost.settled);
      const reserved = number(cost.reserved);
      const meter = el('div', 'cost-meter');
      meter.setAttribute('aria-label', 'Usage estimates and maximum reservations as a share of the approved ceiling');
      const used = el('span', 'settled');
      const held = el('span', 'reserved');
      used.style.width = cap > 0 && settled !== null ? `${Math.min(100, settled / cap * 100)}%` : '0%';
      held.style.width = cap > 0 && reserved !== null ? `${Math.min(100, reserved / cap * 100)}%` : '0%';
      meter.append(used, held);
      const note = cost.note || 'Token-usage estimates are not a reconciled invoice. Unknown charges retain their maximum reservation.';
      section.append(heading, values, meter, el('p', '', note));
      ledger.append(section);
    }
  }

  function renderUpdates(updates) {
    const list = $('update-list');
    list.replaceChildren();
    for (const update of [...(updates || [])].reverse().slice(0, 18)) {
      const row = el('li');
      const time = el('time', '', date(update.time));
      if (update.time) time.dateTime = String(update.time);
      row.append(time, el('span', '', update.text || 'Checkpoint updated'));
      list.append(row);
    }
    if (!list.children.length) list.append(el('li', '', 'No research updates received yet.'));
  }

  function render(data) {
    snapshot = data;
    text('headline', data.headline || 'Experiment status');
    text('summary', data.summary || 'No additional status detail.');
    text('experiment-status', friendly(data.status || 'pending'));
    $('experiment-status').className = 'status-pill' + (/wait|review|approval|blocked/.test(String(data.status)) ? ' waiting' : /fail|error/.test(String(data.status)) ? ' error' : '');
    text('updated-at', `Snapshot ${date(data.updated_at)}`);
    const measured = (data.metrics || []).some(m => number(m.scored) > 0);
    text('evidence-status', measured ? 'Measured outcomes available' : 'No measured model results');
    const d = data.data || {};
    for (const [id, value] of Object.entries({'train-count':d.train, 'dev-count':d.dev, 'train-tokens':d.train_tokens, 'max-tokens':d.max_tokens, 'reviewed-count':d.reviewed, 'rejected-count':d.rejected, 'mmlu-count':d.pilot?.mmlu, 'harmbench-count':d.pilot?.harmbench, 'xstest-count':d.pilot?.xstest, 'reserved-count':d.reserved_final})) text(id, count(value));
    const verified = data.verification || {};
    text('verification', number(verified.tests) !== null ? `${count(verified.tests)} checks ${verified.passed === true ? 'passed' : verified.passed === false ? 'need attention' : 'awaiting verification'}` : 'Verification pending');
    $('verification').className = 'verification' + (verified.passed === true ? ' passed' : verified.passed === false ? ' failed' : '');
    renderArms(data.arms);
    renderProgress(data.progress);
    renderMetrics();
    renderCosts(data.cost);
    renderUpdates(data.updates);
    $('risks').replaceChildren(...(data.risks || ['No limitations recorded yet.']).map(risk => el('li', '', typeof risk === 'string' ? risk : risk.detail || risk.label || 'Unresolved limitation')));
    $('top-links').replaceChildren();
    for (const [key, label] of [['repo','Repository'], ['plan','Plan'], ['linear','Task']]) {
      const raw = data.links?.[key];
      if (!raw) continue;
      try {
        const url = new URL(raw, location.origin);
        if (!['http:','https:'].includes(url.protocol)) continue;
        const link = el('a', '', label);
        link.href = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer';
        $('top-links').append(link);
      } catch { /* Ignore invalid links in an otherwise useful snapshot. */ }
    }
  }

  async function refresh() {
    if (inFlight) return;
    inFlight = true;
    $('refresh').disabled = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch('/api/status', {cache:'no-store',signal:controller.signal,headers:{Accept:'application/json'}});
      if (!response.ok) throw new Error(`Status ${response.status}`);
      const data = await response.json();
      if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('Invalid snapshot');
      const sourceWarning = data.stale === true;
      if (!sourceWarning) {
        render(data);
      } else if (!snapshot) {
        // A partial first response must not turn unavailable sources into zeros.
        const unknownCosts = Object.fromEntries(Object.entries(data.cost || {}).map(([provider, cost]) => [provider, {...cost, settled:null, reserved:null, note:'Source unavailable; amounts are unknown.'}]));
        render({...data, data:{}, metrics:[], cost:unknownCosts, verification:{}, headline:'Some status sources are unavailable', summary:'Waiting for a complete snapshot. Unavailable counts and amounts remain unknown.'});
      }
      receivedAt = new Date();
      $('connection').className = sourceWarning ? 'connection failed' : 'connection connected';
      $('connection').replaceChildren(el('i'), document.createTextNode(sourceWarning ? 'Source warning' : 'Live snapshot'));
      text('refresh-time', `Received ${clockTime(receivedAt)}`);
      text('footer-status', sourceWarning ? 'Connected · Some status sources are incomplete' : 'Connected · Updates every 5 seconds');
      $('stale').hidden = !sourceWarning;
      if (sourceWarning) $('stale').textContent = 'Some status sources could not be read. Showing the last complete snapshot where available; retrying automatically.';
    } catch {
      $('connection').className = 'connection failed';
      $('connection').replaceChildren(el('i'), document.createTextNode(snapshot ? 'Snapshot stale' : 'Connecting'));
      $('stale').hidden = false;
      $('stale').textContent = snapshot ? 'Connection interrupted. Showing the last received snapshot; retrying automatically.' : 'Waiting for the local status service. This page will reconnect automatically.';
      text('footer-status', receivedAt ? `Last received ${clockTime(receivedAt)}` : 'No snapshot received');
    } finally {
      clearTimeout(timeout); inFlight = false; $('refresh').disabled = false;
    }
  }
  $('refresh').addEventListener('click', refresh);
  $('suite-filter').addEventListener('change', renderMetrics);
  $('metric-filter').addEventListener('change', renderMetrics);
  const observer = new IntersectionObserver(entries => {
    const active = entries.filter(entry => entry.isIntersecting).sort((a,b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
    if (active) for (const link of document.querySelectorAll('.rail nav a')) link.classList.toggle('active', link.hash === `#${active.target.id}`);
  }, {rootMargin:'-10% 0px -70% 0px'});
  document.querySelectorAll('main section[id]').forEach(section => observer.observe(section));
  if (window.FIREWORKSHACK_SNAPSHOT) {
    render(window.FIREWORKSHACK_SNAPSHOT);
    $('connection').className = 'connection connected';
    $('connection').replaceChildren(el('i'), document.createTextNode('Saved snapshot'));
    text('refresh-time', `Saved ${clockTime(new Date(window.FIREWORKSHACK_SNAPSHOT.updated_at))}`);
    text('footer-status', 'Saved snapshot · Reload the file for a newer export');
    $('refresh').hidden = true;
  } else {
    refresh();
    setInterval(refresh, 5000);
  }
})();
