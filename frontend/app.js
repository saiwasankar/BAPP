/* ===================================================================
   BAPPS Frontend — Application Logic with Real AI Face Recognition
   & Passport Document Ingestion
   =================================================================== */
const API = 'http://127.0.0.1:8000';

let currentSession = null;
let faceAttempts = [];
let allPassengers = [];
let currentView = 'dashboard';

// Passport file state
let uploadedPassportB64 = null;
let uploadedPassportName = null;

// Camera stream state
let activeModalStream = null;
let activeStudioStream = null;
let modalTrackingTimer = null;
let studioTrackingTimer = null;
let currentFacingMode = 'user';

// ────────────────────────── helpers ──────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(API + path, opts);
  const raw = await res.text();
  let data;
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    data = { detail: raw || `HTTP ${res.status}` };
  }
  if (!res.ok) {
    const msg = data?.error?.message || data?.detail?.[0]?.msg || data?.detail || JSON.stringify(data);
    throw new Error(msg);
  }
  return data;
}

function toast(message, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast toast--${type}`;
  el.textContent = message;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 4500);
}

function faceAnalysisHtml(attempt) {
  const d = attempt?.face_details || {};
  const pct = value => value !== null && value !== undefined && Number.isFinite(Number(value))
    ? `${(Number(value) * 100).toFixed(1)}%` : '—';
  const angles = d.head_angles || {};
  const angleText = ['yaw', 'pitch', 'roll'].some(k => Number.isFinite(Number(angles[k])))
    ? `Yaw ${angles.yaw ?? '—'}° · Pitch ${angles.pitch ?? '—'}° · Roll ${angles.roll ?? '—'}°`
    : '—';
  const diagnosis = attempt?.message || d.guidance || 'No additional diagnosis returned.';
  return `<div class="face-analysis">
    <strong>Scan analysis</strong>
    <span>${escHtml(diagnosis)}</span>
    <span>Liveness ${pct(attempt?.liveness_score)} · Spoof risk ${pct(d.spoof_probability)} · Lighting ${pct(d.lighting_score)}</span>
    <span>Pose ${pct(d.pose_score)} · Eyes open ${pct(d.eye_open_score)} · ${escHtml(angleText)}</span>
  </div>`;
}

function formatTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatDateTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }) + ' ' + formatTime(iso);
}

function stateBadge(state) {
  const map = {
    'AWAITING_FACE':    'blue',
    'FACE_VERIFIED':    'green',
    'CHECKS_COMPLETE':  'green',
    'AWAITING_OFFICER': 'amber',
    'GATE_OPEN':        'green',
    'DENIED':           'red',
    'COMPLETED':        'gray',
  };
  return `<span class="badge badge--${map[state] || 'gray'}">${state}</span>`;
}

function escHtml(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

// ────────────────────────── clock ──────────────────────────
function updateClock() {
  const now = new Date();
  document.getElementById('header-clock').textContent =
    now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}
setInterval(updateClock, 1000);
updateClock();

// ────────────────────────── navigation ──────────────────────────
const tabs = document.querySelectorAll('.nav-tab');
const views = document.querySelectorAll('.view');

function switchView(viewName) {
  currentView = viewName;
  tabs.forEach(t => t.classList.toggle('nav-tab--active', t.dataset.view === viewName));
  views.forEach(v => v.classList.toggle('hidden', v.id !== `view-${viewName}`));
  
  if (viewName !== 'camera' && activeStudioStream) {
    stopStudioCamera();
  }

  // Refresh view data
  if (viewName === 'dashboard')  loadDashboard();
  if (viewName === 'officer')    loadReviews();
  if (viewName === 'camera')     loadCameraStudio();
  if (viewName === 'audit')      loadAudit();
  if (viewName === 'config')     loadConfig();
}

tabs.forEach(t => t.addEventListener('click', () => switchView(t.dataset.view)));

// ────────────────────────── health check ──────────────────────────
async function checkHealth() {
  try {
    await api('GET', '/api/health');
    const el = document.getElementById('connection-status');
    el.innerHTML = '<span class="status-dot status-dot--online"></span>Connected';
    return true;
  } catch {
    const el = document.getElementById('connection-status');
    el.innerHTML = '<span class="status-dot status-dot--offline"></span>Offline';
    return false;
  }
}

// ══════════════════════════════════════════════════════════════
//  DASHBOARD VIEW (ROBUST & AUTO-UPDATING)
// ══════════════════════════════════════════════════════════════
async function loadDashboard() {
  try {
    // Robust independent requests so one failure doesn't block the entire dashboard
    let passengers = [], sessions = [], gates = [], kiosks = [], reviews = [];

    try { passengers = await api('GET', '/api/passengers'); allPassengers = passengers; } catch (e) { console.warn('Passengers load error:', e); }
    try { sessions = await api('GET', '/api/admin/sessions?limit=50'); } catch (e) { console.warn('Sessions load error:', e); }
    try { gates = await api('GET', '/api/egate/gates'); } catch (e) { console.warn('Gates load error:', e); }
    try { kiosks = await api('GET', '/api/egate/kiosks'); } catch (e) { console.warn('Kiosks load error:', e); }
    try { reviews = await api('GET', '/api/immigration/reviews'); } catch (e) { console.warn('Reviews load error:', e); }

    // Stats
    const active = sessions.filter(s => !['COMPLETED', 'DENIED'].includes(s.state)).length;
    const completed = sessions.filter(s => s.state === 'COMPLETED').length;
    const denied = sessions.filter(s => s.state === 'DENIED').length;

    document.getElementById('stat-active').textContent = active;
    document.getElementById('stat-officer').textContent = reviews.length;
    document.getElementById('stat-completed').textContent = completed;
    document.getElementById('stat-denied').textContent = denied;

    // Update officer badge
    const badge = document.getElementById('officer-badge');
    if (reviews.length > 0) {
      badge.textContent = reviews.length;
      badge.style.display = '';
    } else {
      badge.style.display = 'none';
    }

    // Gates
    if (gates.length) {
      document.getElementById('gate-status-list').innerHTML = `<div class="status-card-row">` +
        gates.map(g => `
          <div class="status-card">
            <div class="status-card__icon status-card__icon--${g.status.toLowerCase()}">${g.status === 'OPEN' ? '⬆' : '⬇'}</div>
            <div>
              <div class="status-card__label">${escHtml(g.gate_id)}</div>
              <div class="status-card__sub">${g.status}</div>
            </div>
          </div>`).join('') + `</div>`;
    }

    // Kiosks
    if (kiosks.length) {
      document.getElementById('kiosk-status-list').innerHTML = `<div class="status-card-row">` +
        kiosks.map(k => `
          <div class="status-card">
            <div class="status-card__icon status-card__icon--online">◉</div>
            <div>
              <div class="status-card__label">${escHtml(k.kiosk_id)}</div>
              <div class="status-card__sub">${escHtml(k.location)} · Gate ${escHtml(k.gate_id)} (${k.gate_status})</div>
            </div>
          </div>`).join('') + `</div>`;
    }

    // Passengers Table
    if (passengers.length) {
      const pTbody = document.getElementById('passengers-tbody');
      pTbody.innerHTML = passengers.map(p => `
        <tr style="cursor:pointer;" onclick="selectPassengerForScan('${p.passport_number}', '${p.full_name}')">
          <td><code>${escHtml(p.passenger_id)}</code></td>
          <td><strong>${escHtml(p.full_name)}</strong></td>
          <td>${escHtml(p.nationality)}</td>
          <td><code>${escHtml(p.passport_number)}</code></td>
          <td>${escHtml(p.date_of_birth)}</td>
          <td><span class="badge badge--blue">${escHtml(p.demo_scenario || 'Standard')}</span></td>
        </tr>`).join('');
    }

    // Sessions Table
    const sTbody = document.getElementById('sessions-tbody');
    sTbody.innerHTML = sessions.length ? sessions.slice(0, 15).map(s => `
      <tr style="cursor:pointer;" onclick="loadExistingSession('${s.session_id}')">
        <td><code>${escHtml(s.session_id)}</code></td>
        <td>${escHtml(s.passenger_name)} (${escHtml(s.passenger_id)})</td>
        <td>${stateBadge(s.state)}</td>
        <td>${escHtml(s.gate_id || '—')}</td>
        <td><code>${escHtml(s.next_action)}</code></td>
        <td>${formatTime(s.updated_at)}</td>
      </tr>`).join('') : '<tr><td colspan="6" class="muted">No recent sessions found.</td></tr>';

  } catch (e) {
    console.error('Dashboard load error:', e);
  }
}

document.getElementById('btn-refresh-dashboard').addEventListener('click', loadDashboard);

// Periodic background auto-update for Dashboard
setInterval(() => {
  if (currentView === 'dashboard') {
    loadDashboard();
  }
}, 4000);

window.selectPassengerForScan = function(passport, name) {
  switchView('processing');
  document.getElementById('scan-passport').value = passport;
  document.getElementById('scan-booking').value = '';
  // Auto-attach sample passport for this passenger
  attachSamplePassportFile(passport, name);
};

window.loadExistingSession = async function(sessionId) {
  try {
    const session = await api('GET', `/api/egate/sessions/${sessionId}`);
    currentSession = session;
    faceAttempts = [];
    switchView('processing');
    renderSession();
  } catch (e) {
    toast('Could not load session: ' + e.message, 'error');
  }
};

// ══════════════════════════════════════════════════════════════
//  PASSPORT UPLOAD & DOCUMENT PROCESSING
// ══════════════════════════════════════════════════════════════

// Synthetic Document / Passport Generator (Creates a clear photo portrait document)
function createSamplePassportDocument(name = 'Passenger', passportNo = 'N4821907', seed = 1) {
  const canvas = document.createElement('canvas');
  canvas.width = 440;
  canvas.height = 320;
  const ctx = canvas.getContext('2d');

  // Background booklet
  ctx.fillStyle = '#1E293B';
  ctx.fillRect(0, 0, 440, 320);
  ctx.fillStyle = '#F8FAFC';
  ctx.fillRect(10, 10, 420, 300);

  // Header
  ctx.fillStyle = '#1B2A4A';
  ctx.fillRect(10, 10, 420, 40);
  ctx.fillStyle = '#FFFFFF';
  ctx.font = 'bold 13px Inter, sans-serif';
  ctx.fillText('OFFICIAL PASSPORT / PASSEPORT', 24, 35);

  // Portrait Photo Box (left)
  ctx.fillStyle = '#E2E8F0';
  ctx.fillRect(25, 65, 110, 140);
  ctx.strokeStyle = '#94A3B8';
  ctx.lineWidth = 1;
  ctx.strokeRect(25, 65, 110, 140);

  // Draw face portrait in passport
  const skinTone = seed % 2 === 0 ? '#E5C29F' : '#D2A679';
  ctx.fillStyle = skinTone;
  ctx.beginPath();
  ctx.ellipse(80, 125, 36, 48, 0, 0, Math.PI * 2);
  ctx.fill();

  // Eyes
  ctx.fillStyle = '#1E293B';
  ctx.beginPath();
  ctx.arc(68, 116, 4, 0, Math.PI * 2);
  ctx.arc(92, 116, 4, 0, Math.PI * 2);
  ctx.fill();

  // Nose
  ctx.strokeStyle = '#996633';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(80, 120);
  ctx.lineTo(77, 134);
  ctx.lineTo(83, 134);
  ctx.stroke();

  // Mouth
  ctx.strokeStyle = '#B91C1C';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(80, 145, 12, 0.1 * Math.PI, 0.9 * Math.PI);
  ctx.stroke();

  // Passport Info details (right)
  ctx.fillStyle = '#334155';
  ctx.font = '11px Inter, sans-serif';
  ctx.fillText('SURNAME / NOM', 155, 80);
  ctx.font = 'bold 13px Inter, sans-serif';
  ctx.fillText(name.toUpperCase(), 155, 96);

  ctx.font = '11px Inter, sans-serif';
  ctx.fillText('PASSPORT NO.', 155, 125);
  ctx.font = 'bold 13px monospace';
  ctx.fillText(passportNo, 155, 141);

  ctx.font = '11px Inter, sans-serif';
  ctx.fillText('NATIONALITY', 155, 170);
  ctx.font = 'bold 12px Inter, sans-serif';
  ctx.fillText('UNITED REPUBLIC', 155, 186);

  // Machine Readable Zone (MRZ)
  ctx.fillStyle = '#CBD5E1';
  ctx.fillRect(20, 225, 400, 70);
  ctx.fillStyle = '#0F172A';
  ctx.font = 'bold 11px monospace';
  ctx.fillText(`P<UTO${passportNo.padEnd(9, '<')}0UTO9001014M2801019<<<<<<<<<<<<<<04`, 28, 252);
  ctx.fillText(`${name.replace(/\s+/g, '<<').toUpperCase()}<<<<<<<<<<<<<<<<<<<<<<<<<<`, 28, 274);

  return canvas.toDataURL('image/jpeg', 0.92);
}

function setPassportFile(base64Data, filename = 'passport_document.jpg') {
  uploadedPassportB64 = base64Data;
  uploadedPassportName = filename;

  const dropzonePrompt = document.getElementById('dropzone-prompt');
  const previewCard = document.getElementById('passport-preview-card');
  const previewImg = document.getElementById('passport-preview-img');
  const fileNameEl = document.getElementById('passport-file-name');
  const mandateBadge = document.getElementById('upload-mandate-badge');

  dropzonePrompt.classList.add('hidden');
  previewCard.classList.remove('hidden');
  previewImg.src = base64Data;
  fileNameEl.textContent = filename;

  mandateBadge.className = 'badge badge--green';
  mandateBadge.textContent = '✓ Passport Attached';
  document.getElementById('passport-dropzone').style.borderColor = 'var(--green)';
}

function removePassportFile() {
  uploadedPassportB64 = null;
  uploadedPassportName = null;

  const dropzonePrompt = document.getElementById('dropzone-prompt');
  const previewCard = document.getElementById('passport-preview-card');
  const mandateBadge = document.getElementById('upload-mandate-badge');

  dropzonePrompt.classList.remove('hidden');
  previewCard.classList.add('hidden');
  document.getElementById('passport-file-input').value = '';

  mandateBadge.className = 'badge badge--amber';
  mandateBadge.textContent = 'Passport Document Upload Required';
  document.getElementById('passport-dropzone').style.borderColor = 'var(--gray-300)';
}

document.getElementById('btn-remove-passport').addEventListener('click', (e) => {
  e.stopPropagation();
  removePassportFile();
});

document.getElementById('btn-browse-passport').addEventListener('click', (e) => {
  e.stopPropagation();
  document.getElementById('passport-file-input').click();
});

document.getElementById('passport-dropzone').addEventListener('click', () => {
  if (!uploadedPassportB64) {
    document.getElementById('passport-file-input').click();
  }
});

// File input change handler (User uploads their real file)
document.getElementById('passport-file-input').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(evt) {
    setPassportFile(evt.target.result, file.name);
    toast(`Passport file "${file.name}" uploaded successfully`, 'success');
  };
  reader.readAsDataURL(file);
});

// Drag & Drop handlers
const dropzone = document.getElementById('passport-dropzone');
['dragenter', 'dragover'].forEach(name => {
  dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
});
['dragleave', 'drop'].forEach(name => {
  dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
});
dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(evt) {
    setPassportFile(evt.target.result, file.name);
    toast(`Passport file "${file.name}" attached`, 'success');
  };
  reader.readAsDataURL(file);
});

function attachSamplePassportFile(passportNo, name = 'Passenger') {
  const sampleDocB64 = createSamplePassportDocument(name, passportNo, passportNo.charCodeAt(0));
  setPassportFile(sampleDocB64, `sample_passport_${passportNo}.jpg`);
}

// Quick select buttons (auto attach sample passport image)
document.querySelectorAll('.quick-select button').forEach(btn => {
  btn.addEventListener('click', () => {
    const passportNo = btn.dataset.passport || '';
    const bookingId = btn.dataset.booking || '';
    document.getElementById('scan-passport').value = passportNo;
    document.getElementById('scan-booking').value = bookingId;
    attachSamplePassportFile(passportNo, btn.textContent.split(' ')[1] || 'Traveler');
  });
});

// ══════════════════════════════════════════════════════════════
//  PASSENGER PROCESSING (STEP 1 SCAN & E-GATE)
// ══════════════════════════════════════════════════════════════

document.getElementById('scan-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const passport = document.getElementById('scan-passport').value.trim();
  const booking = document.getElementById('scan-booking').value.trim();
  const kiosk = document.getElementById('scan-kiosk').value;

  // RULE: Until a passport file is uploaded, do NOT allow to pass/scan the passenger!
  if (!uploadedPassportB64) {
    toast('⚠ Upload Required: Please upload your passport document image before scanning.', 'error');
    document.getElementById('passport-dropzone').style.borderColor = 'var(--red)';
    return;
  }

  if (!passport && !booking) {
    toast('Provide a passport number or booking ID.', 'error');
    return;
  }

  try {
    toast('Extracting portrait and verifying travel documents with AI...', 'info');
    const body = {
      kiosk_id: kiosk,
      passport_image_b64: uploadedPassportB64
    };
    if (passport) body.passport_number = passport;
    if (booking) body.booking_id = booking;

    const session = await api('POST', '/api/passengers/scan', body);
    currentSession = session;
    faceAttempts = [];
    renderSession();
    loadDashboard();

    if (session.state === 'AWAITING_FACE') {
      toast(`Session ${session.session_id} ready. Launching live biometric face camera...`, 'info');
      // Automatically pop up the live camera modal to verify real face against uploaded document
      openCameraModal(session);
    } else if (session.state === 'AWAITING_OFFICER') {
      toast(`Session ${session.session_id}: Document verification issue (${session.escalation_reason || 'Check photo'}). Escalated to officer.`, 'error');
    }
  } catch (e) {
    toast('Scan failed: ' + e.message, 'error');
  }
});

// Face verification (Simulated fallback)
document.getElementById('btn-face-verify').addEventListener('click', async () => {
  if (!currentSession) return;
  try {
    const res = await api('POST', `/api/egate/sessions/${currentSession.session_id}/face-verification`);
    currentSession = res.session;
    faceAttempts.push(res.attempt);
    const a = res.attempt;
    if (a.outcome === 'FACE_VERIFIED') toast('Face verified successfully', 'success');
    else if (a.outcome === 'ESCALATED') toast('Max attempts reached — escalated to officer', 'error');
    else toast(`Attempt ${a.attempt_number}: retry (${a.attempts_remaining} remaining)`, 'info');
    renderSession();
    loadDashboard();
  } catch (e) {
    toast('Face verification error: ' + e.message, 'error');
  }
});

// Open Live Camera Popup Button
document.getElementById('btn-open-camera').addEventListener('click', () => {
  if (!currentSession) return;
  openCameraModal(currentSession);
});

// Immigration checks
document.getElementById('btn-run-checks').addEventListener('click', async () => {
  if (!currentSession) return;
  try {
    const res = await api('POST', `/api/immigration/sessions/${currentSession.session_id}/checks`);
    currentSession = res.session;
    const im = res.immigration;
    const wl = res.watchlist;
    let msg = `Immigration: ${im.visa_status}`;
    if (wl.hit) msg += ` | Watchlist: ${wl.highest_risk}`;
    toast(msg, wl.hit ? 'error' : 'success');
    renderSession();
    loadDashboard();
  } catch (e) {
    toast('Check error: ' + e.message, 'error');
  }
});

// Decision
document.getElementById('btn-decision').addEventListener('click', async () => {
  if (!currentSession) return;
  try {
    const res = await api('POST', `/api/egate/sessions/${currentSession.session_id}/decision`);
    currentSession = res.session;
    if (res.authorized) toast('Gate authorized — OPEN', 'success');
    else toast('Escalated to officer: ' + res.reasons.join(', '), 'error');
    renderSession();
    loadDashboard();
  } catch (e) {
    toast('Decision error: ' + e.message, 'error');
  }
});

// Passage
document.getElementById('btn-passage').addEventListener('click', async () => {
  if (!currentSession) return;
  try {
    const res = await api('POST', `/api/egate/sessions/${currentSession.session_id}/passage`);
    currentSession = res.session;
    toast('Passage confirmed — gate closed & session complete', 'success');
    renderSession();
    loadDashboard();
  } catch (e) {
    toast('Passage error: ' + e.message, 'error');
  }
});

// New session
document.getElementById('btn-new-session').addEventListener('click', () => {
  currentSession = null;
  faceAttempts = [];
  document.getElementById('active-session-panel').classList.add('hidden');
  document.getElementById('scan-passport').value = '';
  document.getElementById('scan-booking').value = '';
  removePassportFile();
});

function renderSession() {
  if (!currentSession) return;
  const s = currentSession;

  document.getElementById('active-session-panel').classList.remove('hidden');
  document.getElementById('active-session-id').textContent = s.session_id;
  document.getElementById('active-session-state').className = 'badge';
  document.getElementById('active-session-state').innerHTML = stateBadge(s.state);

  // Pipeline
  updatePipeline(s);

  // Session info
  const info = document.getElementById('session-info');
  const fields = [
    ['Passenger', `${s.passenger_name} (${s.passenger_id})`],
    ['Passport', s.passport_number],
    ['Booking', s.booking_id || '—'],
    ['Kiosk / Gate', `${s.kiosk_id} / ${s.gate_id || '—'}`],
    ['Attempts', `${s.attempts} / ${s.max_attempts}`],
    ['Liveness Score', s.liveness_score !== null ? s.liveness_score.toFixed(4) : '—'],
    ['Match Score', s.match_score !== null ? s.match_score.toFixed(4) : '—'],
    ['Match Passed', s.match_passed !== null ? (s.match_passed ? 'Yes' : 'No') : '—'],
    ['Passport Attached', s.passport_photo_b64 ? 'Portrait Extracted ✓' : 'Standard'],
    ['Immigration', s.immigration_status || '—'],
    ['Entry Permitted', s.entry_permitted !== null ? (s.entry_permitted ? 'Yes' : 'No') : '—'],
    ['Watchlist Hit', s.watchlist_hit !== null ? (s.watchlist_hit ? `Yes (${s.watchlist_risk})` : 'No') : '—'],
    ['Decision', s.decision || '—'],
    ['Escalation', s.escalation_reason || 'None'],
    ['Next Action', s.next_action],
  ];
  if (s.document_issues && s.document_issues.length) {
    fields.push(['Document Issues', s.document_issues.join(', ')]);
  }
  if (s.decision_reasons && s.decision_reasons.length) {
    fields.push(['Decision Reasons', s.decision_reasons.join(', ')]);
  }
  if (s.officer_id) {
    fields.push(['Officer', s.officer_id]);
    fields.push(['Officer Notes', s.officer_notes || '—']);
  }
  info.innerHTML = fields.map(([label, value]) => `
    <div class="session-info__item">
      <span class="session-info__label">${escHtml(label)}</span>
      <span class="session-info__value">${escHtml(String(value))}</span>
    </div>`).join('');

  // Action buttons
  const state = s.state;
  document.getElementById('btn-open-camera').disabled = state !== 'AWAITING_FACE';
  document.getElementById('btn-face-verify').disabled = state !== 'AWAITING_FACE';
  document.getElementById('btn-run-checks').disabled = state !== 'FACE_VERIFIED';
  document.getElementById('btn-decision').disabled = state !== 'CHECKS_COMPLETE';
  document.getElementById('btn-passage').disabled = state !== 'GATE_OPEN';

  // Attempt log
  const logEl = document.getElementById('attempt-log');
  const entriesEl = document.getElementById('attempt-log-entries');
  if (faceAttempts.length) {
    logEl.classList.remove('hidden');
    entriesEl.innerHTML = faceAttempts.map(a => {
      const cls = a.outcome === 'FACE_VERIFIED' ? 'pass' : a.outcome === 'ESCALATED' ? 'escalated' : 'fail';
      const engineTag = a.engine_type === 'AI_REAL_MEDIAPIPE' ? '<span class="badge badge--green">MediaPipe AI (1-to-1 Match)</span>' : '<span class="badge badge--gray">Simulated</span>';
      return `<div class="attempt-entry attempt-entry--${cls}">
        <span class="attempt-entry__num">#${a.attempt_number}</span>
        <span>Liveness: ${a.liveness_passed ? '✓' : '✗'} (${a.liveness_score !== null ? (a.liveness_score * 100).toFixed(1) + '%' : '—'})</span>
        <span>Match: ${a.match_passed === true ? '✓ MATCH' : a.match_passed === false ? '✗ MISMATCH' : '—'} (${a.match_score !== null ? (a.match_score * 100).toFixed(1) + '%' : '—'})</span>
        ${engineTag}
        <span class="badge badge--${cls === 'pass' ? 'green' : cls === 'escalated' ? 'amber' : 'red'}">${a.outcome}</span>
        <span class="muted">${a.attempts_remaining} remaining</span>
        ${faceAnalysisHtml(a)}
      </div>`;
    }).join('');
  } else {
    logEl.classList.add('hidden');
  }
}

function updatePipeline(s) {
  const stepMap = {
    'AWAITING_FACE':    'face',
    'FACE_VERIFIED':    'checks',
    'CHECKS_COMPLETE':  'decision',
    'AWAITING_OFFICER': 'decision',
    'GATE_OPEN':        'passage',
    'DENIED':           'decision',
    'COMPLETED':        'passage',
  };
  const activeStep = stepMap[s.state] || 'scan';
  const order = ['scan', 'face', 'checks', 'decision', 'passage'];
  const activeIdx = order.indexOf(activeStep);

  document.querySelectorAll('.pipeline__step').forEach((el) => {
    const step = el.dataset.step;
    const idx = order.indexOf(step);
    el.classList.remove('pipeline__step--active', 'pipeline__step--done', 'pipeline__step--failed');

    if (s.state === 'DENIED' && step === 'decision') {
      el.classList.add('pipeline__step--failed');
    } else if (s.state === 'AWAITING_OFFICER' && step === 'decision') {
      el.classList.add('pipeline__step--active');
    } else if (idx < activeIdx || s.state === 'COMPLETED') {
      el.classList.add('pipeline__step--done');
    } else if (idx === activeIdx) {
      el.classList.add('pipeline__step--active');
    }
  });
}

// ══════════════════════════════════════════════════════════════
//  CAMERA ENGINE & POPUP MODAL (1-TO-1 PASSPORT VS LIVE SCAN)
// ══════════════════════════════════════════════════════════════

async function startWebcam(videoEl) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: currentFacingMode,
      },
      audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play();
    return stream;
  } catch (err) {
    console.warn('Webcam permission error or no camera attached:', err);
    toast('Camera notice: ' + err.message + ' (You can also use "Use Clean Test Sample")', 'info');
    return null;
  }
}

function stopWebcam(videoEl, stream) {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
  if (videoEl) {
    videoEl.srcObject = null;
  }
}

function captureVideoFrame(videoEl) {
  // Never substitute the enrolled document portrait for a live camera frame.
  if (!videoEl || videoEl.videoWidth === 0 || videoEl.videoHeight === 0 || !videoEl.srcObject) {
    return null;
  }
  const canvas = document.createElement('canvas');
  canvas.width = videoEl.videoWidth || 640;
  canvas.height = videoEl.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', 0.88);
}

function drawBiometricHud(canvasEl, videoEl, det) {
  if (!canvasEl || !videoEl) return;
  const w = videoEl.videoWidth || canvasEl.offsetWidth || 640;
  const h = videoEl.videoHeight || canvasEl.offsetHeight || 480;
  canvasEl.width = w;
  canvasEl.height = h;
  const ctx = canvasEl.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  if (!det || !det.face_detected || !det.bbox) return;

  const b = det.bbox;
  const bx = b.norm_x * w;
  const by = b.norm_y * h;
  const bw = b.norm_w * w;
  const bh = b.norm_h * h;

  // Draw bounding box
  const isPass = det.liveness_passed;
  ctx.strokeStyle = isPass ? '#22C55E' : '#EAB308';
  ctx.lineWidth = 2.5;
  ctx.strokeRect(bx, by, bw, bh);

  // Corner brackets
  const cornerLen = 14;
  ctx.strokeStyle = '#FFFFFF';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(bx, by + cornerLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cornerLen, by);
  ctx.moveTo(bx + bw - cornerLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + cornerLen);
  ctx.moveTo(bx, by + bh - cornerLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + cornerLen, by + bh);
  ctx.moveTo(bx + bw - cornerLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - cornerLen);
  ctx.stroke();

  // Face label
  ctx.fillStyle = isPass ? 'rgba(34, 197, 94, 0.85)' : 'rgba(234, 179, 8, 0.85)';
  ctx.fillRect(bx, Math.max(0, by - 22), 170, 20);
  ctx.fillStyle = '#FFFFFF';
  ctx.font = 'bold 11px Inter, sans-serif';
  ctx.fillText(`AI LIVENESS: ${det.liveness_score ? (det.liveness_score * 100).toFixed(1) + '%' : 'TRACKING'}`, bx + 6, Math.max(14, by - 8));
}

// ────────────────── CAMERA MODAL (SIDE-BY-SIDE VERIFICATION) ──────────────────

async function openCameraModal(session) {
  const modal = document.getElementById('camera-modal');
  const video = document.getElementById('modal-video');
  const canvas = document.getElementById('modal-canvas');
  const guidance = document.getElementById('modal-guidance');
  const oval = modal.querySelector('.hud-target-oval');
  const verdictBanner = document.getElementById('modal-verdict-banner');

  verdictBanner.className = 'match-verdict-banner hidden';
  verdictBanner.innerHTML = '';

  modal.classList.remove('hidden');
  document.getElementById('modal-session-tag').textContent = `SESSION: ${session.session_id}`;
  document.getElementById('modal-pax-name').textContent = session.passenger_name;
  document.getElementById('modal-pax-passport').textContent = session.passport_number;
  document.getElementById('modal-pax-attempts').textContent = `${session.attempts} / ${session.max_attempts}`;
  document.getElementById('modal-ref-id').textContent = `Passport: ${session.passport_number}`;

  // Side-by-Side: Load Uploaded Passport Portrait
  const passportImg = document.getElementById('modal-passport-img');
  if (session.passport_photo_b64) {
    passportImg.src = session.passport_photo_b64;
    document.getElementById('modal-ref-badge').textContent = 'Extracted Passport Portrait';
  } else if (uploadedPassportB64) {
    passportImg.src = uploadedPassportB64;
    document.getElementById('modal-ref-badge').textContent = 'Attached Passport Document';
  } else {
    // Generate default visual card
    passportImg.src = createSamplePassportDocument(session.passenger_name, session.passport_number);
    document.getElementById('modal-ref-badge').textContent = 'Biometric Reference Ready';
  }

  // Start webcam
  activeModalStream = await startWebcam(video);

  // Real-time HUD tracking loop
  if (modalTrackingTimer) clearInterval(modalTrackingTimer);
  modalTrackingTimer = setInterval(async () => {
    if (modal.classList.contains('hidden')) {
      clearInterval(modalTrackingTimer);
      return;
    }
    const frame = captureVideoFrame(video);
    if (!frame) {
      document.getElementById('modal-liveness-text').textContent = 'Camera unavailable — allow camera access and retry';
      document.getElementById('modal-guidance').textContent = 'A live camera feed is required to analyze your face.';
      document.getElementById('modal-m-liveness').textContent = '—';
      document.getElementById('modal-m-spoof').textContent = '—';
      return;
    }
    try {
      const det = await api('POST', '/api/egate/live-face-detect', { image_base64: frame });
      drawBiometricHud(canvas, video, det);

      const dot = document.getElementById('modal-dot');
      const text = document.getElementById('modal-liveness-text');
      const livenessScore = det.liveness_score !== null ? (det.liveness_score * 100).toFixed(1) + '%' : '—';
      
      document.getElementById('modal-m-liveness').textContent = livenessScore;
      document.getElementById('modal-bar-liveness').style.width = det.liveness_score ? `${det.liveness_score * 100}%` : '0%';
      document.getElementById('modal-m-spoof').textContent = det.spoof_probability !== null && det.spoof_probability !== undefined
        ? `${(det.spoof_probability * 100).toFixed(1)}%` : '—';
      document.getElementById('modal-landmarks-badge').textContent = `${det.landmarks_count || 0} / 478 3D Landmarks`;

      if (det.head_angles) {
        document.getElementById('modal-m-pose').textContent = `Yaw: ${det.head_angles.yaw}°, Pitch: ${det.head_angles.pitch}°`;
      }
      if (det.eye_open_score !== null) {
        document.getElementById('modal-m-eyes').textContent = `${(det.eye_open_score * 100).toFixed(0)}% Open`;
      }

      if (det.face_detected) {
        dot.className = det.liveness_passed ? 'status-dot status-dot--online' : 'status-dot';
        dot.style.background = det.liveness_passed ? '#22C55E' : '#EAB308';
        text.textContent = det.liveness_passed ? `Live Face Confirmed (${livenessScore})` : `Analyzing Liveness (${livenessScore})`;
        guidance.textContent = det.guidance || 'Face aligned';
        oval.className = det.liveness_passed ? 'hud-target-oval aligned' : 'hud-target-oval';
      } else {
        dot.className = 'status-dot status-dot--offline';
        dot.style.background = '#94A3B8';
        text.textContent = 'Waiting for face alignment...';
        guidance.textContent = 'Please position your face within the frame';
        oval.className = 'hud-target-oval';
      }
    } catch (e) {
      // background poll ignore
    }
  }, 350);
}

function closeCameraModal() {
  const modal = document.getElementById('camera-modal');
  const video = document.getElementById('modal-video');
  if (modalTrackingTimer) clearInterval(modalTrackingTimer);
  stopWebcam(video, activeModalStream);
  activeModalStream = null;
  modal.classList.add('hidden');
}

document.getElementById('camera-modal-close').addEventListener('click', closeCameraModal);

// Modal Capture & Verify Button (Non-hallucinating verification decision)
document.getElementById('btn-modal-capture').addEventListener('click', async () => {
  if (!currentSession) return;
  const video = document.getElementById('modal-video');
  const frame = captureVideoFrame(video);
  const verdictBanner = document.getElementById('modal-verdict-banner');
  if (!frame) {
    toast('A working camera is required for live face verification.', 'error');
    return;
  }

  try {
    toast('Executing MediaPipe AI biometric matching against uploaded passport...', 'info');
    const res = await api('POST', `/api/egate/sessions/${currentSession.session_id}/face-verification`, {
      image_base64: frame,
      frame_id: 'CAM-' + Date.now(),
    });

    currentSession = res.session;
    faceAttempts.push(res.attempt);
    const a = res.attempt;
    const matchPct = a.match_score !== null && a.match_score !== undefined
      ? (a.match_score * 100).toFixed(1) + '%' : '—';
    const livenessPct = a.liveness_score !== null && a.liveness_score !== undefined
      ? (a.liveness_score * 100).toFixed(1) + '%' : '—';

    verdictBanner.classList.remove('hidden');

    if (a.match_passed) {
      verdictBanner.className = 'match-verdict-banner match-verdict-banner--success';
      verdictBanner.innerHTML = `
        <strong>✓ BIOMETRIC MATCH VERIFIED (${matchPct})</strong><br>
        Live facial scan matches the uploaded passport photo.<br>
        <span style="font-size:0.75rem;opacity:0.9;">Liveness: ${livenessPct} · 478 3D Landmarks verified · Identity Confirmed</span>
        ${faceAnalysisHtml(a)}
      `;
      toast(`✓ Biometric Verified! Match Confidence: ${matchPct}`, 'success');
      setTimeout(() => {
        closeCameraModal();
        renderSession();
        loadDashboard();
      }, 1400);
    } else if (!a.liveness_passed) {
      verdictBanner.className = 'match-verdict-banner match-verdict-banner--mismatch';
      verdictBanner.innerHTML = `
        <strong>✗ LIVE FACE ANALYSIS FAILED</strong><br>
        ${escHtml(a.message || 'The camera frame could not be analyzed. Check camera access and AI model availability.')}
        <span style="font-size:0.75rem;opacity:0.9;">Attempt ${a.attempt_number} of 3 · ${a.attempts_remaining} attempts remaining</span>
        ${faceAnalysisHtml(a)}
      `;
      toast(a.message || 'Live face analysis failed. Check the camera and try again.', 'error');
      if (a.outcome === 'ESCALATED') {
        setTimeout(() => {
          closeCameraModal();
          renderSession();
          loadDashboard();
        }, 1800);
      }
    } else {
      // MISMATCH: State clearly with exact score
      verdictBanner.className = 'match-verdict-banner match-verdict-banner--mismatch';
      verdictBanner.innerHTML = `
        <strong>✗ BIOMETRIC MISMATCH DETECTED (${matchPct} < 85.0%)</strong><br>
        The live facial scan DOES NOT MATCH the uploaded passport photo.<br>
        <span style="font-size:0.75rem;opacity:0.9;">Attempt ${a.attempt_number} of 3 · ${a.attempts_remaining} attempts remaining</span>
        ${faceAnalysisHtml(a)}
      `;
      toast(`✗ Biometric Mismatch (${matchPct}): Live face does not match passport.`, 'error');
      if (a.outcome === 'ESCALATED') {
        toast('Max biometric attempts exhausted — Session escalated to immigration officer.', 'error');
        setTimeout(() => {
          closeCameraModal();
          renderSession();
          loadDashboard();
        }, 1800);
      }
    }

    renderSession();
    loadDashboard();
  } catch (e) {
    toast('Capture failed: ' + e.message, 'error');
  }
});

// Modal Enrol Face Template Button
document.getElementById('btn-modal-enroll').addEventListener('click', async () => {
  if (!currentSession) return;
  const video = document.getElementById('modal-video');
  const frame = captureVideoFrame(video);
  try {
    const res = await api('POST', `/api/egate/passengers/${currentSession.passenger_id}/enroll-face`, {
      image_base64: frame
    });
    toast(res.message, 'success');
  } catch (e) {
    toast('Enrollment failed: ' + e.message, 'error');
  }
});

// Modal Use Clean Test Sample Button
document.getElementById('btn-modal-test-frame').addEventListener('click', async () => {
  if (!currentSession) return;
  const frame = uploadedPassportB64 || createSamplePassportDocument(currentSession.passenger_name, currentSession.passport_number);
  try {
    const res = await api('POST', `/api/egate/sessions/${currentSession.session_id}/face-verification`, {
      image_base64: frame,
      frame_id: 'SYNTH-' + Date.now(),
    });
    currentSession = res.session;
    faceAttempts.push(res.attempt);
    toast('Sample matched: ' + res.attempt.outcome + ` (${(res.attempt.match_score * 100).toFixed(1)}%)`, 'success');
    if (res.attempt.outcome === 'FACE_VERIFIED') closeCameraModal();
    renderSession();
    loadDashboard();
  } catch (e) {
    toast('Test failed: ' + e.message, 'error');
  }
});

// ────────────────── CAMERA STUDIO VIEW ──────────────────

async function loadCameraStudio() {
  try {
    const passengers = allPassengers.length ? allPassengers : await api('GET', '/api/passengers');
    allPassengers = passengers;
    const select = document.getElementById('studio-passenger-select');
    select.innerHTML = '<option value="">-- Select Passenger to Match --</option>' +
      passengers.map(p => `<option value="${p.passenger_id}">${p.full_name} (${p.passenger_id} - ${p.passport_number})</option>`).join('');

    startStudioCamera();
  } catch (e) {
    toast('Failed to load studio passengers: ' + e.message, 'error');
  }
}

async function startStudioCamera() {
  const video = document.getElementById('studio-video');
  const canvas = document.getElementById('studio-canvas');
  const guidance = document.getElementById('studio-guidance');
  const oval = document.getElementById('camera-studio-stage').querySelector('.hud-target-oval');

  activeStudioStream = await startWebcam(video);

  document.getElementById('btn-studio-start').classList.add('hidden');
  document.getElementById('btn-studio-stop').classList.remove('hidden');
  document.getElementById('btn-studio-scan-now').disabled = false;
  document.getElementById('btn-studio-match-passenger').disabled = false;
  document.getElementById('btn-studio-enroll-passenger').disabled = false;

  if (studioTrackingTimer) clearInterval(studioTrackingTimer);
  studioTrackingTimer = setInterval(async () => {
    if (!document.getElementById('chk-auto-scan').checked) return;
    const frame = captureVideoFrame(video);
    if (!frame) {
      document.getElementById('telem-status').textContent = 'CAMERA UNAVAILABLE';
      guidance.textContent = 'Start the camera and allow camera access to analyze a face.';
      return;
    }
    try {
      const det = await api('POST', '/api/egate/live-face-detect', { image_base64: frame });
      drawBiometricHud(canvas, video, det);

      document.getElementById('telem-status').textContent = det.face_detected ? (det.liveness_passed ? 'LIVE FACE DETECTED' : 'UNALIGNED / CHECK') : 'NO FACE DETECTED';
      document.getElementById('telem-landmarks').textContent = det.landmarks_count ? `${det.landmarks_count} (3D Mesh)` : '0';
      document.getElementById('telem-liveness').textContent = det.liveness_score !== null ? `${(det.liveness_score * 100).toFixed(1)}%` : '—';
      document.getElementById('telem-depth').textContent = det.depth_score !== null ? `${(det.depth_score * 100).toFixed(0)}% (3D Structure)` : '—';
      document.getElementById('telem-eyes').textContent = det.eye_open_score !== null ? `${(det.eye_open_score * 100).toFixed(0)}% (Open)` : '—';
      
      if (det.head_angles) {
        document.getElementById('telem-pose').textContent = `Yaw: ${det.head_angles.yaw}°, Pitch: ${det.head_angles.pitch}°`;
      }
      document.getElementById('telem-lighting').textContent = det.lighting_score !== null ? `${(det.lighting_score * 100).toFixed(0)}% Quality` : '—';

      guidance.textContent = det.guidance || 'Position face in oval';
      oval.className = det.liveness_passed ? 'hud-target-oval aligned' : 'hud-target-oval';
    } catch (e) {
      // background poll ignore
    }
  }, 350);
}

function stopStudioCamera() {
  const video = document.getElementById('studio-video');
  if (studioTrackingTimer) clearInterval(studioTrackingTimer);
  stopWebcam(video, activeStudioStream);
  activeStudioStream = null;

  document.getElementById('btn-studio-start').classList.remove('hidden');
  document.getElementById('btn-studio-stop').classList.add('hidden');
  document.getElementById('btn-studio-scan-now').disabled = true;
  document.getElementById('btn-studio-match-passenger').disabled = true;
  document.getElementById('btn-studio-enroll-passenger').disabled = true;
  document.getElementById('telem-status').textContent = 'STANDBY';
}

document.getElementById('btn-studio-start').addEventListener('click', startStudioCamera);
document.getElementById('btn-studio-stop').addEventListener('click', stopStudioCamera);

document.getElementById('btn-camera-switch').addEventListener('click', async () => {
  currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
  if (activeStudioStream) {
    stopStudioCamera();
    startStudioCamera();
  }
});

document.getElementById('btn-studio-scan-now').addEventListener('click', async () => {
  const video = document.getElementById('studio-video');
  const frame = captureVideoFrame(video);
  try {
    const det = await api('POST', '/api/egate/live-face-detect', { image_base64: frame });
    toast(`AI Face Scan: ${det.face_detected ? 'Face Detected (Liveness: ' + (det.liveness_score * 100).toFixed(1) + '%)' : 'No Face'}`, det.face_detected ? 'success' : 'error');
  } catch (e) {
    toast('Scan error: ' + e.message, 'error');
  }
});

document.getElementById('btn-studio-match-passenger').addEventListener('click', async () => {
  const paxId = document.getElementById('studio-passenger-select').value;
  if (!paxId) {
    toast('Please select a passenger to match against.', 'error');
    return;
  }
  const video = document.getElementById('studio-video');
  const frame = captureVideoFrame(video);
  const resultBox = document.getElementById('studio-match-result');

  try {
    const res = await api('POST', `/api/egate/passengers/${paxId}/enroll-face`, { image_base64: frame });
    resultBox.classList.remove('hidden', 'match-result-box--error');
    resultBox.classList.add('match-result-box--success');
    resultBox.innerHTML = `<strong>Result for ${paxId}:</strong><br>${escHtml(res.message)}`;
    toast('Face comparison completed', 'success');
  } catch (e) {
    resultBox.classList.remove('hidden', 'match-result-box--success');
    resultBox.classList.add('match-result-box--error');
    resultBox.innerHTML = `<strong>Match Failed:</strong><br>${escHtml(e.message)}`;
    toast('Match failed: ' + e.message, 'error');
  }
});

document.getElementById('btn-studio-enroll-passenger').addEventListener('click', async () => {
  const paxId = document.getElementById('studio-passenger-select').value;
  if (!paxId) {
    toast('Please select a passenger to enroll.', 'error');
    return;
  }
  const video = document.getElementById('studio-video');
  const frame = captureVideoFrame(video);
  try {
    const res = await api('POST', `/api/egate/passengers/${paxId}/enroll-face`, { image_base64: frame });
    toast(`Enrollment updated for ${paxId}: ${res.message}`, 'success');
  } catch (e) {
    toast('Enrollment failed: ' + e.message, 'error');
  }
});

const studioOpenModalBtn = document.getElementById('btn-studio-open-modal');
if (studioOpenModalBtn) {
  studioOpenModalBtn.addEventListener('click', () => {
    if (currentSession) {
      openCameraModal(currentSession);
    } else {
      const paxId = document.getElementById('studio-passenger-select').value || 'P-1001';
      const pax = allPassengers.find(p => p.passenger_id === paxId) || allPassengers[0];
      const demoSession = {
        session_id: 'STUDIO-LIVE-CAPTURE',
        passenger_id: pax ? pax.passenger_id : 'P-1001',
        passenger_name: pax ? pax.full_name : 'Passenger',
        passport_number: pax ? pax.passport_number : 'N4821907',
        attempts: 0,
        max_attempts: 3,
        passport_photo_b64: uploadedPassportB64 || null
      };
      openCameraModal(demoSession);
    }
  });
}

// ══════════════════════════════════════════════════════════════
//  OFFICER REVIEW VIEW
// ══════════════════════════════════════════════════════════════
async function loadReviews() {
  try {
    const reviews = await api('GET', '/api/immigration/reviews');
    const badge = document.getElementById('officer-badge');
    if (reviews.length > 0) {
      badge.textContent = reviews.length;
      badge.style.display = '';
    } else {
      badge.style.display = 'none';
    }

    const list = document.getElementById('reviews-list');
    if (!reviews.length) {
      list.innerHTML = '<p class="muted">No sessions currently awaiting officer review.</p>';
      return;
    }

    list.innerHTML = reviews.map(r => `
      <div class="review-card">
        <div class="review-card__header">
          <div>
            <strong>${escHtml(r.passenger_name)}</strong>
            <span class="muted" style="margin-left:8px;">${escHtml(r.passenger_id)}</span>
          </div>
          <span class="badge badge--amber">${escHtml(r.escalation_reason || 'ESCALATED')}</span>
        </div>
        <div class="review-card__body">
          <div class="review-card__field"><span class="review-card__label">Session:</span> <code>${escHtml(r.session_id)}</code></div>
          <div class="review-card__field"><span class="review-card__label">Passport:</span> <code>${escHtml(r.passport_number)}</code></div>
          <div class="review-card__field"><span class="review-card__label">Kiosk / Gate:</span> ${escHtml(r.kiosk_id)} / ${escHtml(r.gate_id || '—')}</div>
          <div class="review-card__field"><span class="review-card__label">Liveness:</span> ${r.liveness_score !== null ? (r.liveness_score * 100).toFixed(1) + '%' : '—'}</div>
          <div class="review-card__field"><span class="review-card__label">Match Score:</span> ${r.match_score !== null ? (r.match_score * 100).toFixed(1) + '%' : '—'}</div>
          <div class="review-card__field"><span class="review-card__label">Watchlist Hit:</span> ${r.watchlist_hit ? `Yes (${r.watchlist_risk})` : 'No'}</div>
          <div class="review-card__field"><span class="review-card__label">Entry Permitted:</span> ${r.entry_permitted !== null ? (r.entry_permitted ? 'Yes' : 'No') : '—'}</div>
          <div class="review-card__field"><span class="review-card__label">Issues:</span> ${(r.document_issues || []).join(', ') || 'None'}</div>
        </div>
        <div class="action-bar" style="margin-top:12px;">
          <button class="btn btn--primary btn--sm" onclick="openOverrideModal('${r.session_id}')">Open Review</button>
        </div>
      </div>`).join('');
  } catch (e) {
    toast('Reviews load failed: ' + e.message, 'error');
  }
}

document.getElementById('btn-refresh-reviews').addEventListener('click', loadReviews);

// Override modal
let overrideSessionId = null;

window.openOverrideModal = async function(sessionId) {
  overrideSessionId = sessionId;
  try {
    const s = await api('GET', `/api/egate/sessions/${sessionId}`);
    document.getElementById('override-session-id').textContent = s.session_id;
    const info = document.getElementById('override-session-info');
    info.innerHTML = `
      <div class="session-info__item"><span class="session-info__label">Passenger</span><span class="session-info__value">${escHtml(s.passenger_name)} (${escHtml(s.passenger_id)})</span></div>
      <div class="session-info__item"><span class="session-info__label">Passport</span><span class="session-info__value">${escHtml(s.passport_number)}</span></div>
      <div class="session-info__item"><span class="session-info__label">Escalation Reason</span><span class="session-info__value" style="color:var(--amber);font-weight:600;">${escHtml(s.escalation_reason || '—')}</span></div>
      <div class="session-info__item"><span class="session-info__label">Document Issues</span><span class="session-info__value">${(s.document_issues || []).join(', ') || 'None'}</span></div>
      <div class="session-info__item"><span class="session-info__label">Match Score</span><span class="session-info__value">${s.match_score !== null ? (s.match_score * 100).toFixed(1) + '%' : '—'}</span></div>
      <div class="session-info__item"><span class="session-info__label">Watchlist</span><span class="session-info__value">${s.watchlist_hit ? `HIT: ${s.watchlist_risk}` : 'Clear'}</span></div>
    `;
    document.getElementById('override-notes').value = '';
    document.getElementById('override-modal').classList.remove('hidden');
  } catch (e) {
    toast('Could not load session for review: ' + e.message, 'error');
  }
};

function closeOverrideModal() {
  document.getElementById('override-modal').classList.add('hidden');
  overrideSessionId = null;
}

document.getElementById('override-close').addEventListener('click', closeOverrideModal);

async function submitOverride(approve) {
  if (!overrideSessionId) return;
  const officerId = document.getElementById('override-officer').value;
  const notes = document.getElementById('override-notes').value.trim();

  if (notes.length < 10) {
    toast('Justification notes must be at least 10 characters long.', 'error');
    return;
  }

  try {
    const res = await api('POST', `/api/immigration/sessions/${overrideSessionId}/override`, {
      officer_id: officerId,
      approve,
      notes,
    });
    toast(`Officer override recorded: ${approve ? 'APPROVED (gate open)' : 'DENIED'}`, approve ? 'success' : 'error');
    closeOverrideModal();
    loadReviews();
    loadDashboard();
    if (currentSession && currentSession.session_id === overrideSessionId) {
      currentSession = res.session;
      renderSession();
    }
  } catch (e) {
    toast('Override failed: ' + e.message, 'error');
  }
}

document.getElementById('btn-override-approve').addEventListener('click', () => submitOverride(true));
document.getElementById('btn-override-deny').addEventListener('click', () => submitOverride(false));

// ══════════════════════════════════════════════════════════════
//  AUDIT LOG VIEW
// ══════════════════════════════════════════════════════════════
let auditOffset = 0;
const auditLimit = 25;

async function loadAudit(offset = 0) {
  auditOffset = offset;
  try {
    const session = document.getElementById('audit-session').value.trim();
    const event = document.getElementById('audit-event').value.trim();
    const actor = document.getElementById('audit-actor').value.trim();

    const params = new URLSearchParams({ limit: auditLimit, offset });
    if (session) params.set('session_id', session);
    if (event)   params.set('event_type', event);
    if (actor)   params.set('actor_id', actor);

    const data = await api('GET', `/api/admin/audit-log?${params.toString()}`);
    const tbody = document.getElementById('audit-tbody');

    tbody.innerHTML = data.items.length
      ? data.items.map(e => `
        <tr>
          <td>${e.log_id}</td>
          <td>${formatDateTime(e.timestamp)}</td>
          <td>${e.session_id ? `<code>${escHtml(e.session_id)}</code>` : '—'}</td>
          <td><span class="badge badge--blue">${escHtml(e.event_type)}</span></td>
          <td>${escHtml(e.actor_type)}: <strong>${escHtml(e.actor_id)}</strong></td>
          <td>${e.decision ? stateBadge(e.decision) : '—'}</td>
          <td class="muted" style="max-width:220px;font-size:0.75rem;word-break:break-word;">${escHtml(JSON.stringify(e.details))}</td>
        </tr>`).join('')
      : '<tr><td colspan="7" class="muted">No audit entries found.</td></tr>';

    const pag = document.getElementById('audit-pagination');
    const total = data.total;
    const showing = Math.min(offset + auditLimit, total);
    pag.innerHTML = `
      <span>Showing ${total > 0 ? offset + 1 : 0}–${showing} of ${total}</span>
      ${offset > 0 ? `<button class="btn btn--outline btn--xs" onclick="loadAudit(${Math.max(0, offset - auditLimit)})">◀ Previous</button>` : ''}
      ${showing < total ? `<button class="btn btn--outline btn--xs" onclick="loadAudit(${offset + auditLimit})">Next ▶</button>` : ''}
    `;
  } catch (e) {
    toast('Audit load failed: ' + e.message, 'error');
  }
}

window.loadAudit = loadAudit;

document.getElementById('audit-filter-form').addEventListener('submit', (e) => {
  e.preventDefault();
  loadAudit(0);
});
document.getElementById('btn-audit-clear').addEventListener('click', () => {
  document.getElementById('audit-session').value = '';
  document.getElementById('audit-event').value = '';
  document.getElementById('audit-actor').value = '';
  loadAudit(0);
});
document.getElementById('btn-refresh-audit').addEventListener('click', () => loadAudit(auditOffset));

// Verify chain
document.getElementById('btn-verify-chain').addEventListener('click', async () => {
  try {
    const res = await api('GET', '/api/admin/audit-log/verify');
    const el = document.getElementById('chain-status');
    el.classList.remove('hidden', 'chain-status--valid', 'chain-status--invalid');
    if (res.valid) {
      el.classList.add('chain-status--valid');
      el.textContent = `✓ Chain valid — ${res.entries_checked} entries verified. ${res.message}`;
    } else {
      el.classList.add('chain-status--invalid');
      el.textContent = `✗ Chain INVALID at log #${res.first_invalid_log_id}. ${res.message}`;
    }
  } catch (e) {
    toast('Verification failed: ' + e.message, 'error');
  }
});

// ══════════════════════════════════════════════════════════════
//  SYSTEM CONFIG VIEW
// ══════════════════════════════════════════════════════════════
async function loadConfig() {
  try {
    const cfg = await api('GET', '/api/admin/config');
    document.getElementById('cfg-match').textContent = cfg.match_threshold.toFixed(2);
    document.getElementById('cfg-liveness').textContent = cfg.liveness_threshold.toFixed(2);
    document.getElementById('cfg-attempts').textContent = cfg.max_attempts;
  } catch (e) {
    toast('Config load failed: ' + e.message, 'error');
  }
}

document.getElementById('config-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const body = { admin_id: 'ADM-001' };
  const match = document.getElementById('cfg-new-match').value;
  const liveness = document.getElementById('cfg-new-liveness').value;
  const attempts = document.getElementById('cfg-new-attempts').value;

  if (match) body.match_threshold = parseFloat(match);
  if (liveness) body.liveness_threshold = parseFloat(liveness);
  if (attempts) body.max_attempts = parseInt(attempts);

  if (!match && !liveness && !attempts) {
    toast('Enter at least one value to update.', 'error');
    return;
  }

  try {
    await api('PUT', '/api/admin/config', body);
    toast('Configuration updated', 'success');
    loadConfig();
    document.getElementById('cfg-new-match').value = '';
    document.getElementById('cfg-new-liveness').value = '';
    document.getElementById('cfg-new-attempts').value = '';
  } catch (e) {
    toast('Config update failed: ' + e.message, 'error');
  }
});

// Demo reset
document.getElementById('btn-reset-db').addEventListener('click', async () => {
  if (!confirm('This will erase ALL sessions, audit log entries, and configuration changes.\n\nAre you sure?')) return;
  try {
    await api('POST', '/api/admin/demo/reset');
    toast('Database reset and re-seeded', 'success');
    currentSession = null;
    faceAttempts = [];
    document.getElementById('active-session-panel').classList.add('hidden');
    removePassportFile();
    loadDashboard();
    loadConfig();
  } catch (e) {
    toast('Reset failed: ' + e.message, 'error');
  }
});

// ══════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════
(async function init() {
  const ok = await checkHealth();
  if (ok) {
    loadDashboard();
  } else {
    toast('Cannot connect to BAPPS backend at ' + API + '. Is uvicorn running?', 'error');
  }
  setInterval(checkHealth, 15000);
})();
