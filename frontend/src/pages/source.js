import {api, download} from '../api/client.js';
import {empty, errorSlot} from '../components/common/layout.js';
import {table} from '../components/common/table.js';
import {actions} from '../constants/enums.js';
import {bindForm} from '../hooks/forms.js';
import {state} from '../stores/session.js';
import {esc,time} from '../utils/html.js';

export async function overview() {
  const profile = state.profile;
  const history = await api('/audit');
  return `<section class="summary"><h2>${esc(profile.organization)}</h2><p>${esc(profile.context)}</p><ol>${profile.scenario.map(step => `<li>${esc(step)}</li>`).join('')}</ol><a class="button" href="#1">核对数据</a></section><section><h2>最近操作</h2>${table(history.map(row => ({时间:time(row.at),账号:row.actor,操作:actions[row.action] ?? row.action,记录:row.subject})))}</section>`;
}

export async function source() {
  const profile = state.profile;
  const revisions = await api('/datasets');
  const active = revisions.find(row => row.id === state.datasetId) || revisions[0];
  const detail = active ? await api(`/datasets/${active.id}`) : null;
  const importer = state.account.role === 'analyst' ? `<form id="upload" class="inline-form"><label>导入测量记录<input name="csv" type="file" accept=".csv,text/csv" required></label><label>导入站点与批次对应表<input name="batches" type="file" accept=".csv,text/csv" required></label><button type="submit">保存数据版本</button>${errorSlot}</form>` : '';
  const samples = `<button type="button" id="example" class="secondary">下载样例测量记录</button> <button type="button" id="example-batches" class="secondary">下载样例批次对应表</button>`;
  const summary = detail ? `<p class="hint">${detail.count} 行测量记录 · ${detail.batch_count} 个校准批次 · 页面预览前 ${detail.records.length} 行 · SHA-256 ${esc(detail.checksum.slice(0,12))}</p>` : '';
  const batches = detail && detail.batches.length ? `<h3>站点与校准批次对应</h3>${table(detail.batches,profile.batch_columns)}` : '';
  return `<section><p>${esc(profile.input)}</p><p class="hint">两份 CSV 的表头顺序可调整，列名需匹配样例。测量记录与批次对应表须一次同时提交，少配或多配都会退回。</p>${importer}${samples}<p id="download-error" role="alert" class="error"></p></section>${detail ? `<section><label>数据版本<select id="revision">${revisions.map(row => `<option value="${row.id}" ${row.id === active.id ? 'selected' : ''}>${esc(row.name)} · #${row.id}</option>`).join('')}</select></label>${summary}${table(detail.records,profile.columns)}${batches}</section>` : empty('请导入项目样例或业务 CSV。')}`;
}

export function bindSource(root, refresh) {
  bindForm(root, '#upload', async form => {
    const csv = form.elements.csv.files[0];
    const batches = form.elements.batches.files[0];
    const imported = await api('/datasets', {name:csv.name,csv:await csv.text(),batches:await batches.text()});
    state.datasetId = imported.id;
    state.status = imported.created ? `已保存 ${imported.rows} 行测量记录、${imported.stations} 座站点归属。` : '数据与已有版本一致，已选中该版本。';
  }, refresh);
  root.querySelector('#revision')?.addEventListener('change', event => {
    state.datasetId = Number(event.target.value);
    refresh();
  });
  root.querySelector('#example')?.addEventListener('click', async () => {
    try {await download('/example.csv', state.profile.filename);}
    catch (error) {root.querySelector('#download-error').textContent = error.message;}
  });
  root.querySelector('#example-batches')?.addEventListener('click', async () => {
    try {await download('/example-batches.csv', state.profile.batch_filename);}
    catch (error) {root.querySelector('#download-error').textContent = error.message;}
  });
}
