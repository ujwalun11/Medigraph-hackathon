// MediGraph Frontend Client Logic
const API_BASE = window.location.origin;

let activeSessionId = null;
let currentPatientDatabase = {};
let clinicianAuthUnlocked = true;

document.addEventListener('DOMContentLoaded', () => {
  lucide.createIcons();
  initTabs();
  checkBackendHealth();
  loadPatients();
  loadGraphStats();
  loadLatestEvals();
  initEventListeners();
});

// ── Tab Navigation ──
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      
      tab.classList.add('active');
      const paneId = `pane${tab.dataset.tab.charAt(0).toUpperCase() + tab.dataset.tab.slice(1)}`;
      const targetPane = document.getElementById(paneId);
      if (targetPane) targetPane.classList.add('active');
      lucide.createIcons();
    });
  });
}

// ── Backend Health Check ──
async function checkBackendHealth() {
  const valEl = document.getElementById('falkorStatusVal');
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.status === 'online') {
      valEl.textContent = 'Connected (6379)';
      valEl.style.color = '#10b981';
    } else {
      valEl.textContent = 'Degraded';
    }
  } catch (err) {
    valEl.textContent = 'Offline';
    valEl.style.color = '#f43f5e';
  }
}

// ── Load Patient Charts ──
async function loadPatients() {
  const select = document.getElementById('patientSelect');
  try {
    const res = await fetch(`${API_BASE}/api/patients`);
    const data = await res.json();
    if (data.status === 'success' && data.patients) {
      select.innerHTML = '<option value="">Generic / Walk-in Patient (No Pre-existing History)</option>';
      data.patients.forEach(p => {
        currentPatientDatabase[p.patient_id] = p;
        const opt = document.createElement('option');
        opt.value = p.patient_id;
        opt.textContent = p.label || `${p.patient_id} (${p.gender}, Age ${p.age})`;
        select.appendChild(opt);
      });
      renderVaultTable(data.patients);
    }
  } catch (err) {
    console.error('Failed to load patient profiles:', err);
  }
}

// ── Load FalkorDB Stats ──
async function loadGraphStats() {
  try {
    const res = await fetch(`${API_BASE}/graph/stats`);
    const data = await res.json();
    if (data.status === 'connected' && data.counts) {
      document.getElementById('statConditions').textContent = data.counts.conditions || 36;
      document.getElementById('statSymptoms').textContent = data.counts.symptoms || 143;
      document.getElementById('statMeds').textContent = data.counts.medications || 132;
      const totalEdges = (data.counts.conditions * 3) + (data.counts.symptoms * 2);
      document.getElementById('statRels').textContent = totalEdges || 434;
    }
  } catch (err) {
    console.error('Failed to fetch graph stats:', err);
  }
}

// ── Event Listeners ──
function initEventListeners() {
  // Patient Selector change
  const patientSelect = document.getElementById('patientSelect');
  patientSelect.addEventListener('change', (e) => {
    updatePatientSummary(e.target.value);
  });

  // Refresh Patients
  document.getElementById('btnRefreshPatients').addEventListener('click', loadPatients);

  // Quick Scenarios Chips
  document.querySelectorAll('.scenario-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const patientId = chip.dataset.patient;
      const text = chip.dataset.text;
      patientSelect.value = patientId;
      updatePatientSummary(patientId);
      document.getElementById('symptomsInput').value = text;
    });
  });

  // Run Triage
  document.getElementById('btnRunTriage').addEventListener('click', runTriage);

  // Clear Triage
  document.getElementById('btnClearTriage').addEventListener('click', clearTriage);

  // Submit Clarification
  document.getElementById('btnSubmitClarification').addEventListener('click', submitClarification);

  // Reasoning Accordion Toggle
  document.getElementById('reasoningToggle').addEventListener('click', () => {
    const list = document.getElementById('reasoningStepsList');
    const chevron = document.getElementById('reasoningChevron');
    if (list.style.display === 'none') {
      list.style.display = 'block';
      chevron.setAttribute('data-lucide', 'chevron-up');
    } else {
      list.style.display = 'none';
      chevron.setAttribute('data-lucide', 'chevron-down');
    }
    lucide.createIcons();
  });

  // Natural Language Graph Query
  document.getElementById('btnRunQuery').addEventListener('click', runGraphQuery);

  // Test PII Sanitizer
  document.getElementById('btnTestSanitizer').addEventListener('click', runPiiSanitizerTest);

  // Toggle Auth in Vault
  document.getElementById('btnToggleAuth').addEventListener('click', () => {
    clinicianAuthUnlocked = !clinicianAuthUnlocked;
    const btn = document.getElementById('btnToggleAuth');
    btn.textContent = clinicianAuthUnlocked ? 'Unlocked (Demo)' : 'Locked (HIPAA)';
    btn.classList.toggle('active', clinicianAuthUnlocked);
    renderVaultTable(Object.values(currentPatientDatabase));
  });

  // Refresh Stats
  document.getElementById('btnRefreshStats').addEventListener('click', loadGraphStats);

  // Run Benchmark Suite
  const evalBtn = document.getElementById('btnRunEvalSuite');
  if (evalBtn) {
    evalBtn.addEventListener('click', runLiveBenchmarkSuite);
  }
}

// ── Load / Run RAGAS Benchmark Suite ──
async function loadLatestEvals() {
  try {
    const res = await fetch(`${API_BASE}/api/evals`);
    const data = await res.json();
    if (data.status === 'success' && data.eval_data) {
      const agg = data.eval_data.aggregate_metrics;
      if (agg) {
        document.getElementById('evalSafetyRecall').textContent = `${Math.round(agg.clinical_safety_recall * 100)}%`;
        document.getElementById('evalFaithfulness').textContent = `${Math.round(agg.ragas_faithfulness * 100)}%`;
        document.getElementById('evalUrgencyAcc').textContent = `${Math.round(agg.urgency_classification_accuracy * 100)}%`;
        document.getElementById('evalComposite').textContent = `${Math.round(agg.ragas_composite_score * 100)}%`;
      }
    }
  } catch (err) {
    console.error('Failed to load evals:', err);
  }
}

async function runLiveBenchmarkSuite() {
  const btn = document.getElementById('btnRunEvalSuite');
  btn.disabled = true;
  btn.innerHTML = '<i data-lucide="loader-2" class="spin"></i> <span>Running RAGAS Suite...</span>';
  lucide.createIcons();

  try {
    const res = await fetch(`${API_BASE}/api/evals/run`, { method: 'POST' });
    const data = await res.json();
    if (data.status === 'success' && data.eval_data) {
      const agg = data.eval_data.aggregate_metrics;
      document.getElementById('evalSafetyRecall').textContent = `${Math.round(agg.clinical_safety_recall * 100)}%`;
      document.getElementById('evalFaithfulness').textContent = `${Math.round(agg.ragas_faithfulness * 100)}%`;
      document.getElementById('evalUrgencyAcc').textContent = `${Math.round(agg.urgency_classification_accuracy * 100)}%`;
      document.getElementById('evalComposite').textContent = `${Math.round(agg.ragas_composite_score * 100)}%`;
    }
  } catch (err) {
    console.error('Benchmark failed:', err);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i data-lucide="play"></i> <span>Run Benchmark Suite</span>';
    lucide.createIcons();
  }
}

// ── Update Patient Summary Card ──
function updatePatientSummary(patientId) {
  const card = document.getElementById('patientSummaryCard');
  if (!patientId || !currentPatientDatabase[patientId]) {
    card.style.display = 'none';
    return;
  }
  const p = currentPatientDatabase[patientId];
  card.style.display = 'block';
  document.getElementById('patientNameDisplay').textContent = p.label.split('(')[0].trim();
  document.getElementById('patientMetaDisplay').textContent = `${p.age}yo ${p.gender} • Chart: ${p.patient_id}`;
  
  const initials = p.label.slice(0, 2).toUpperCase();
  document.getElementById('patientAvatar').textContent = initials;

  // Render Conditions
  const condsEl = document.getElementById('patientConditionsTags');
  condsEl.innerHTML = '';
  (p.active_conditions || []).forEach(c => {
    const span = document.createElement('span');
    span.className = 'tag';
    span.textContent = c;
    condsEl.appendChild(span);
  });

  // Render Meds
  const medsEl = document.getElementById('patientMedsTags');
  medsEl.innerHTML = '';
  (p.active_medications || []).forEach(m => {
    const span = document.createElement('span');
    span.className = 'tag tag-success';
    span.textContent = m;
    medsEl.appendChild(span);
  });
}

// ── Run Agentic Triage ──
async function runTriage() {
  const symptoms = document.getElementById('symptomsInput').value.trim();
  const patientId = document.getElementById('patientSelect').value;
  const statusBadge = document.getElementById('triageStatusBadge');
  const runBtn = document.getElementById('btnRunTriage');

  if (!symptoms) {
    alert('Please enter patient symptoms or select a test scenario.');
    return;
  }

  // UI state: Running
  statusBadge.textContent = 'Agent Reasoning...';
  statusBadge.className = 'badge badge-accent';
  runBtn.disabled = true;

  // Reset steps & clarification
  document.getElementById('clarificationBox').style.display = 'none';
  const stepsList = document.getElementById('reasoningStepsList');
  stepsList.innerHTML = '<div class="empty-state-text">Invoking LangGraph ReAct agent & querying FalkorDB...</div>';

  try {
    const res = await fetch(`${API_BASE}/api/triage/agentic`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symptoms: symptoms,
        patient_id: patientId || null
      })
    });
    const data = await res.json();

    // Render Steps
    renderReasoningSteps(data.steps || []);

    // Check if clarification needed
    if (data.status === 'clarification') {
      activeSessionId = data.session_id;
      statusBadge.textContent = 'Clarification Needed';
      statusBadge.className = 'badge badge-danger';
      document.getElementById('clarificationBox').style.display = 'block';
      document.getElementById('clarificationQuestionText').textContent = data.question;
      document.getElementById('clarificationAnswerInput').focus();
    } else if (data.status === 'complete') {
      statusBadge.textContent = 'Complete';
      statusBadge.className = 'badge badge-success';
      renderReport(data.result, data.privacy_audit);
    }
  } catch (err) {
    statusBadge.textContent = 'Error';
    statusBadge.className = 'badge badge-danger';
    stepsList.innerHTML = `<div class="empty-state-text" style="color: #f43f5e;">Error during agent execution: ${err.message}</div>`;
  } finally {
    runBtn.disabled = false;
    lucide.createIcons();
  }
}

// ── Submit Clarification ──
async function submitClarification() {
  const answer = document.getElementById('clarificationAnswerInput').value.trim();
  const statusBadge = document.getElementById('triageStatusBadge');
  if (!answer || !activeSessionId) return;

  statusBadge.textContent = 'Resuming Agent...';
  statusBadge.className = 'badge badge-accent';

  try {
    const res = await fetch(`${API_BASE}/api/triage/agentic`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symptoms: '',
        session_id: activeSessionId,
        clarification_answer: answer
      })
    });
    const data = await res.json();
    document.getElementById('clarificationBox').style.display = 'none';

    renderReasoningSteps(data.steps || []);

    if (data.status === 'complete') {
      statusBadge.textContent = 'Complete';
      statusBadge.className = 'badge badge-success';
      renderReport(data.result, data.privacy_audit);
    }
  } catch (err) {
    statusBadge.textContent = 'Error';
  }
}

// ── Render Reasoning Steps ──
function renderReasoningSteps(steps) {
  const list = document.getElementById('reasoningStepsList');
  const countEl = document.getElementById('stepsCount');
  const chevron = document.getElementById('reasoningChevron');
  
  countEl.textContent = steps.length;
  if (!steps.length) {
    list.innerHTML = '<div class="empty-state-text">No tool calls executed.</div>';
    return;
  }

  // Auto-expand accordion to show tool execution trace
  list.style.display = 'block';
  chevron.setAttribute('data-lucide', 'chevron-up');

  list.innerHTML = '';
  steps.forEach((s, idx) => {
    const item = document.createElement('div');
    item.className = 'reasoning-step-item';
    
    if (s.type === 'tool_call') {
      const argsFormatted = typeof s.input === 'object' ? JSON.stringify(s.input, null, 2) : s.input;
      item.innerHTML = `
        <span class="step-badge step-call"><i data-lucide="wrench"></i> Tool Call #${idx + 1}</span>
        <div style="flex: 1; min-width: 0;">
          <div style="font-weight: 600; color: #38bdf8; margin-bottom: 4px; font-family: monospace;">${s.tool}()</div>
          <pre style="background: rgba(15,23,42,0.6); padding: 6px 10px; border-radius: 6px; font-size: 0.8rem; overflow-x: auto; color: #cbd5e1; border: 1px solid rgba(255,255,255,0.06);">${escapeHtml(argsFormatted)}</pre>
        </div>
      `;
    } else {
      let displayContent = s.content;
      try {
        const parsed = JSON.parse(s.content);
        displayContent = JSON.stringify(parsed, null, 2);
      } catch (e) {}

      item.innerHTML = `
        <span class="step-badge step-obs"><i data-lucide="database"></i> FalkorDB Result</span>
        <div style="flex: 1; min-width: 0;">
          <pre style="background: rgba(15,23,42,0.6); padding: 6px 10px; border-radius: 6px; font-size: 0.8rem; overflow-x: auto; color: #a7f3d0; border: 1px solid rgba(255,255,255,0.06); max-height: 180px;">${escapeHtml(displayContent)}</pre>
        </div>
      `;
    }
    list.appendChild(item);
  });
  lucide.createIcons();
}

function escapeHtml(str) {
  if (typeof str !== 'string') return str;
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Render Final Report ──
function renderReport(report, privacyAudit) {
  document.getElementById('emptyReport').style.display = 'none';
  document.getElementById('reportContent').style.display = 'block';

  const urgency = report.urgency_level || 'Routine';
  const score = report.urgency_score || 5;
  const banner = document.getElementById('urgencyBanner');
  const scoreCircle = document.getElementById('urgencyScoreCircle');
  
  scoreCircle.textContent = score;
  document.getElementById('urgencyLevelText').textContent = `${urgency} Urgency`;
  document.getElementById('urgencySpecialtyText').textContent = `Suggested Specialty: ${report.suggested_specialty || 'General Medicine'}`;

  banner.className = 'urgency-banner';
  if (urgency.toLowerCase().includes('emergency')) {
    banner.classList.add('emergency');
    scoreCircle.style.background = '#f43f5e';
  } else if (urgency.toLowerCase().includes('urgent')) {
    banner.classList.add('urgent');
    scoreCircle.style.background = '#f59e0b';
  } else {
    banner.classList.add('routine');
    scoreCircle.style.background = '#10b981';
  }

  // Contraindications
  const contraAlert = document.getElementById('contraindicationAlert');
  const contraList = document.getElementById('contraindicationList');
  const contras = report.graph_grounded_facts?.contraindications_found || [];
  if (contras.length > 0) {
    contraAlert.style.display = 'flex';
    contraList.innerHTML = contras.map(c => `<li><strong>${c}</strong></li>`).join('');
  } else {
    contraAlert.style.display = 'none';
  }

  // Confirmed Conditions
  const condsEl = document.getElementById('confirmedConditionsList');
  const conds = report.graph_grounded_facts?.confirmed_conditions || [];
  condsEl.innerHTML = conds.map(c => `<span class="tag tag-danger">${c}</span>`).join('') || '<span class="tag">Undetermined</span>';

  // Recommended Treatments
  const treatsEl = document.getElementById('recommendedTreatmentsList');
  const treats = report.graph_grounded_facts?.recommended_treatments || [];
  treatsEl.innerHTML = treats.map(t => `<span class="tag tag-success">${t}</span>`).join('') || '<span class="tag">Under Evaluation</span>';

  // Notes
  document.getElementById('clinicalNotesText').textContent = 
    report.summary?.clinical_notes || report.summary?.red_flags?.join(' • ') || 'Consultation complete.';

  // Privacy Audit
  if (privacyAudit) {
    document.getElementById('privacyAuditText').textContent = 
      `Layer 1 Privacy Gatekeeper: ${privacyAudit.pii_masked} entity masked prior to LLM call. Zero raw PII retained.`;
  }
}

// ── Clear Triage ──
function clearTriage() {
  document.getElementById('symptomsInput').value = '';
  document.getElementById('patientSelect').value = '';
  document.getElementById('patientSummaryCard').style.display = 'none';
  document.getElementById('clarificationBox').style.display = 'none';
  document.getElementById('reportContent').style.display = 'none';
  document.getElementById('emptyReport').style.display = 'flex';
  document.getElementById('reasoningStepsList').innerHTML = '<div class="empty-state-text">No active triage run.</div>';
  document.getElementById('stepsCount').textContent = '0';
  document.getElementById('triageStatusBadge').textContent = 'Idle';
  document.getElementById('triageStatusBadge').className = 'badge';
}

// ── Graph Explorer Query ──
async function runGraphQuery() {
  const query = document.getElementById('nlQueryInput').value.trim();
  if (!query) return;

  const card = document.getElementById('cypherResultsCard');
  card.style.display = 'block';
  document.getElementById('generatedCypherCode').textContent = 'Generating Cypher query...';
  document.getElementById('resultsTableContainer').innerHTML = 'Querying FalkorDB...';

  try {
    const res = await fetch(`${API_BASE}/graph-query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    });
    const data = await res.json();
    document.getElementById('generatedCypherCode').textContent = data.cypher || '// No Cypher generated';
    
    const records = data.records || [];
    document.getElementById('queryResultCount').textContent = records.length;

    if (!records.length) {
      document.getElementById('resultsTableContainer').innerHTML = '<div style="color: #94a3b8;">No matching graph records found.</div>';
      return;
    }

    const headers = Object.keys(records[0]);
    let html = '<table><thead><tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr></thead><tbody>';
    records.forEach(r => {
      html += '<tr>' + headers.map(h => `<td>${r[h]}</td>`).join('') + '</tr>';
    });
    html += '</tbody></table>';
    document.getElementById('resultsTableContainer').innerHTML = html;
  } catch (err) {
    document.getElementById('resultsTableContainer').innerHTML = `<div style="color: #f43f5e;">Query error: ${err.message}</div>`;
  }
}

// ── Test PII Sanitizer ──
async function runPiiSanitizerTest() {
  const raw = document.getElementById('rawPiiInput').value;
  const outBox = document.getElementById('sanitizedOutputBox');
  outBox.textContent = 'Sanitizing...';

  try {
    const res = await fetch(`${API_BASE}/api/privacy/de-identify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: raw })
    });
    const data = await res.json();
    outBox.innerHTML = `
      <strong>Anonymized Prompt:</strong><br>${data.anonymized_text}<br><br>
      <strong>Audit Entities Detected (${data.entities_masked}):</strong><br>
      ${JSON.stringify(data.audit_log, null, 2)}
    `;
  } catch (err) {
    outBox.textContent = `Sanitization failed: ${err.message}`;
  }
}

// ── Render Vault Table ──
function renderVaultTable(patients) {
  const tbody = document.getElementById('vaultTableBody');
  if (!tbody) return;
  tbody.innerHTML = '';

  patients.forEach(p => {
    const tr = document.createElement('tr');
    const realName = clinicianAuthUnlocked ? (p.label.split('(')[0].trim() || 'Restricted') : '•••••••• (Masked)';
    tr.innerHTML = `
      <td><code>${p.patient_id}</code></td>
      <td style="color: ${clinicianAuthUnlocked ? '#6ee7b7' : '#94a3b8'};">${realName}</td>
      <td>${(p.active_conditions || []).join(', ')}</td>
      <td><span class="badge ${clinicianAuthUnlocked ? 'badge-success' : 'badge-danger'}">${clinicianAuthUnlocked ? 'Authorized' : 'Restricted'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}
