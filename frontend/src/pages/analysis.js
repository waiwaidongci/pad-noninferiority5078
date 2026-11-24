import {api} from '../api/client.js';
import {empty,errorSlot} from '../components/common/layout.js';
import {report} from '../components/common/report.js';
import {table} from '../components/common/table.js';
import {decisions,states} from '../constants/enums.js';
import {bindForm} from '../hooks/forms.js';
import {state} from '../stores/session.js';
import {fieldInput} from '../types/fields.js';
import {esc,formData,time} from '../utils/html.js';

export async function analysis() {
  const profile = state.profile;
  const [datasets,runs] = await Promise.all([api('/datasets'),api('/runs')]);
  const form = state.account.role === 'analyst' && datasets.length ? `<form id="queue" class="inline-form"><label>数据版本<select name="dataset_id">${datasets.map(row => `<option value="${row.id}" ${row.id === state.datasetId ? 'selected' : ''}>${esc(row.name)} · #${row.id}</option>`).join('')}</select></label>${profile.options.map(field => `<label>${esc(field.label)}${fieldInput(field,esc)}</label>`).join('')}<button type="submit">提交计算</button>${errorSlot}</form>` : empty(datasets.length ? '分析员可提交计算。' : '先导入数据版本。', '#1', '查看数据台账');
  return `<section><h2>计算条件</h2><p>${esc(profile.method)}</p>${form}</section><section><h2>分析记录</h2>${table(runs.map(run => ({任务:{link:`#3/${run.id}`,text:`#${run.id}`},数据版本:run.dataset_id,状态:states[run.state],提交时间:time(run.created),复核:decisions[run.decision] ?? '待复核'})))}</section>`;
}

export async function currentRun() {
  const runs = await api('/runs');
  const chosen = runs.find(item => item.id === state.runId) || runs[0];
  return {runs,run:chosen ? await api(`/runs/${chosen.id}`) : null};
}

export function chooseRun(runs, selected) {
  return `<label>分析任务<select id="pick-run">${runs.map(row => `<option value="${row.id}" ${selected === row.id ? 'selected' : ''}>#${row.id} · ${states[row.state]} · ${time(row.created)}</option>`).join('')}</select></label>`;
}

export async function results() {
  const {runs,run} = await currentRun();
  if (!run) return empty('提交一条分析任务后可在这里查看结果。','#2','配置分析');
  const result = run.result ? report(run.result) : `<p ${run.error ? 'class="error"' : 'data-poll'}>${esc(run.error || states[run.state])}</p>`;
  const summary = run.result ? topBound(run.result) : '';
  return `<section>${chooseRun(runs,run.id)}<p class="hint">数据版本 #${run.dataset_id} · 计算状态 ${states[run.state]}</p>${summary}${result}${run.result ? `<a class="button" href="#4/${run.id}">进入业务复核</a>` : '<a href="#2">回到分析条件</a>'}</section>`;
}

function topBound(result) {
  const metrics = result.metrics || {};
  const upper = metrics['最高单侧95%置信上界（kW）'];
  const withdrawn = metrics['最高上界对应撤回批次'];
  const flagged = metrics['触发暂缓的组合数'];
  if (upper == null || withdrawn == null) return `<p class="notice">没有任何撤回组合能算出上界，暂缓验收。</p>`;
  const pairs = metrics['有效配对数'];
  const margin = metrics['预登记非劣效界值（kW）'];
  const verdict = !(upper < margin)
    ? `<p class="notice">最高上界 ${esc(upper)} kW 已达到或超过 ${esc(margin)} kW 界值，撤回校准批次后复核不通过，暂缓验收。</p>`
    : flagged > 0 ? `<p class="notice">最高上界 ${esc(upper)} kW 虽低于 ${esc(margin)} kW 界值，但仍有撤回组合因站点不足或上界算不出而暂缓，验收一并暂缓。</p>` : '';
  return `<section class="summary"><p>本批有效配对 ${esc(pairs)} 组，撤回 0 至 ${esc(metrics['撤回批次上限'] ?? 4)} 批校准批次后，最高单侧 95% 置信上界为 <strong>${esc(upper)} kW</strong>，对应撤回 <strong>${esc(withdrawn)}</strong>。</p>${flagged > 0 ? `<p class="notice">共 ${esc(flagged)} 个撤回组合触发暂缓，已在下方「触发暂缓的组合」中列出。</p>` : ''}${verdict}</section>`;
}

export function bindAnalysis(root, section, refresh) {
  bindForm(root, '#queue', async form => {
    const {dataset_id,...options} = formData(form);
    const run = await api('/runs',{dataset_id:Number(dataset_id),options});
    location.hash = `3/${run.id}`;
  });
  root.querySelector('#pick-run')?.addEventListener('change', event => {location.hash = `${section}/${event.target.value}`;});
}
