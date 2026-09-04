// DOM Elements
const articleInput = document.getElementById('articleInput');
const wordCount = document.getElementById('wordCount');
const charCount = document.getElementById('charCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const sampleBtn = document.getElementById('sampleBtn');
const clearBtn = document.getElementById('clearBtn');
const copyResultBtn = document.getElementById('copyResultBtn');
const signalList = document.getElementById('signalList');
const metricAccuracy = document.getElementById('metricAccuracy');
const metricPrecision = document.getElementById('metricPrecision');
const metricRecall = document.getElementById('metricRecall');
const metricF1 = document.getElementById('metricF1');
const modelSampleCount = document.getElementById('modelSampleCount');
const historyList = document.getElementById('historyList');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const historyCount = document.getElementById('historyCount');
const navButtons = document.querySelectorAll('.nav-btn');
const panels = document.querySelectorAll('.panel');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultContent = document.getElementById('resultContent');

// Iconic HUD Elements
const hudScoreWrapper = document.getElementById('hudScoreWrapper');
const hudAmbientGlow = document.getElementById('hudAmbientGlow');
const hudProgressRing = document.getElementById('hudProgressRing');
const hudStatusIcon = document.getElementById('hudStatusIcon');
const hudIconBadge = document.getElementById('hudIconBadge');
const hudVerdictPill = document.getElementById('hudVerdictPill');
const scorePercentage = document.getElementById('scorePercentage');
const scoreLabelText = document.getElementById('scoreLabelText');
const scoreSummaryText = document.getElementById('scoreSummaryText');
const hudSummaryCard = document.getElementById('hudSummaryCard');
const hudSummaryBadge = document.getElementById('hudSummaryBadge');
const hudConfidenceTag = document.getElementById('hudConfidenceTag');
const reliablePatternBar = document.getElementById('reliablePatternBar');
const misleadingPatternBar = document.getElementById('misleadingPatternBar');
const reliablePatternValue = document.getElementById('reliablePatternValue');
const misleadingPatternValue = document.getElementById('misleadingPatternValue');
const reliableSubtext = document.getElementById('reliableSubtext');
const misleadingSubtext = document.getElementById('misleadingSubtext');

let sampleArticles = [];
let currentAnalysisData = null;
let scoreAnimationTimer = null;

// Local Storage Key for User Browser Isolation
const LOCAL_STORAGE_HISTORY_KEY = 'authentiq_user_local_history';

// ==========================================
// Local Storage Helper Utilities
// ==========================================
function getLocalHistory() {
  try {
    const raw = localStorage.getItem(LOCAL_STORAGE_HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error('Error reading local history:', e);
    return [];
  }
}

function saveLocalHistory(items) {
  try {
    localStorage.setItem(LOCAL_STORAGE_HISTORY_KEY, JSON.stringify(items));
  } catch (e) {
    console.error('Error saving local history:', e);
  }
}

function addLocalHistoryRecord(text, result) {
  const items = getLocalHistory();
  const newRecord = {
    id: 'hist_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7),
    created_at: new Date().toISOString(),
    article: text,
    label: result.label,
    confidence: result.confidence,
    reliability: result.reliability,
    signals: result.signals,
    metrics: result.metrics,
  };
  items.unshift(newRecord);
  saveLocalHistory(items);
  return newRecord;
}

// ==========================================
// Initialization & Startup
// ==========================================
document.addEventListener('DOMContentLoaded', async () => {
  updateTextStats();
  await loadModelInfo();
  await loadSamples();
  loadHistory();
  initEventListeners();
});

// ==========================================
// Event Listeners Initialization
// ==========================================
function initEventListeners() {
  if (articleInput) {
    articleInput.addEventListener('input', updateTextStats);
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', analyzeArticle);
  }

  if (sampleBtn) {
    sampleBtn.addEventListener('click', loadRandomSample);
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', clearInput);
  }

  if (copyResultBtn) {
    copyResultBtn.addEventListener('click', copyCurrentAnalysis);
  }

  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', clearAllHistory);
  }

  // Navigation Panel Switching
  navButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-target');
      setPanel(target);
    });
  });

  // Sample pill buttons
  document.querySelectorAll('.sample-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      const idx = parseInt(pill.getAttribute('data-sample-idx'), 10);
      if (sampleArticles && sampleArticles[idx]) {
        articleInput.value = sampleArticles[idx].text;
        updateTextStats();
        setPanel('analyzer');
      }
    });
  });
}

// ==========================================
// Panel Switching Logic
// ==========================================
function setPanel(targetId) {
  navButtons.forEach((btn) => {
    const isMatch = btn.getAttribute('data-target') === targetId;
    btn.classList.toggle('active', isMatch);
  });

  panels.forEach((p) => {
    const isMatch = p.id === targetId;
    p.classList.toggle('active', isMatch);
  });

  if (targetId === 'history') {
    loadHistory();
  }
}

// ==========================================
// Text Statistics Counter
// ==========================================
function updateTextStats() {
  if (!articleInput) return;
  const val = articleInput.value.trim();
  const words = val ? val.split(/\s+/).filter(Boolean).length : 0;
  const chars = articleInput.value.length;

  if (wordCount) wordCount.textContent = words;
  if (charCount) charCount.textContent = chars;
}

// ==========================================
// Load Model Metrics & Information
// ==========================================
async function loadModelInfo() {
  try {
    const res = await fetch('/api/model-info');
    if (!res.ok) return;
    const data = await res.json();

    if (metricAccuracy) metricAccuracy.textContent = `${(data.metrics.accuracy * 100).toFixed(1)}%`;
    if (metricPrecision) metricPrecision.textContent = `${(data.metrics.precision * 100).toFixed(1)}%`;
    if (metricRecall) metricRecall.textContent = `${(data.metrics.recall * 100).toFixed(1)}%`;
    if (metricF1) metricF1.textContent = `${(data.metrics.f1 * 100).toFixed(1)}%`;

    if (modelSampleCount) {
      modelSampleCount.textContent = `${data.total_samples} Balanced Real-World Messages`;
    }
  } catch (err) {
    console.error('Model info fetch error:', err);
  }
}

// ==========================================
// Load Sample Data
// ==========================================
async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    if (!res.ok) return;
    sampleArticles = await res.json();
  } catch (err) {
    console.error('Sample fetch error:', err);
  }
}

// ==========================================
// Random Sample Generator
// ==========================================
function loadRandomSample() {
  if (!sampleArticles || sampleArticles.length === 0) return;
  const idx = Math.floor(Math.random() * sampleArticles.length);
  const sample = sampleArticles[idx];

  if (articleInput) {
    articleInput.value = sample.text;
    updateTextStats();
    showNotification(`Loaded sample: ${sample.title}`, 'info');
  }
}

// ==========================================
// Clear Input Area
// ==========================================
function clearInput() {
  if (articleInput) {
    articleInput.value = '';
    updateTextStats();
    if (resultContent) resultContent.style.display = 'none';
    currentAnalysisData = null;
  }
}

// ==========================================
// Submit Article for Analysis
// ==========================================
async function analyzeArticle() {
  if (!articleInput) return;
  const text = articleInput.value.trim();

  if (!text) {
    showNotification('Please enter some text to analyze.', 'error');
    return;
  }

  if (text.length < 10) {
    showNotification('Text is too short. Please provide at least 10 characters.', 'error');
    return;
  }

  // Show Loading UI
  if (loadingSpinner) loadingSpinner.style.display = 'flex';
  if (resultContent) resultContent.style.display = 'none';
  if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="spinner-small"></span> Quantum Scanning...';
  }

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Failed to analyze text.');
    }

    currentAnalysisData = data;
    addLocalHistoryRecord(text, data);
    updatePredictionDisplay(data);
    loadHistory();
    showNotification('Intelligence analysis complete!', 'success');
  } catch (error) {
    showNotification(error.message || 'Error occurred during analysis.', 'error');
  } finally {
    if (loadingSpinner) loadingSpinner.style.display = 'none';
    if (resultContent) resultContent.style.display = 'block';
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = '<span class="btn-icon">⚡</span> Analyze Article';
    }
  }
}

// ==========================================
// Animate Score Counter (Rolling Odometer)
// ==========================================
function animateScoreCounter(targetScore, duration = 850) {
  if (!scorePercentage) return;
  if (scoreAnimationTimer) cancelAnimationFrame(scoreAnimationTimer);

  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1.0);
    const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
    const currentVal = (targetScore * easeProgress).toFixed(1);

    scorePercentage.textContent = `${currentVal}%`;

    if (progress < 1.0) {
      scoreAnimationTimer = requestAnimationFrame(update);
    } else {
      scorePercentage.textContent = `${targetScore.toFixed(1)}%`;
    }
  }

  scoreAnimationTimer = requestAnimationFrame(update);
}

// ==========================================
// Update Result Display (Iconic HUD)
// ==========================================
function updatePredictionDisplay(result) {
  const isReliable = result.label === 'Reliable';
  const reliableScore = Number(result.reliable_score ?? 50);
  const misleadingScore = Number(result.misleading_score ?? 50);
  const mainScore = isReliable ? reliableScore : misleadingScore;

  // 1. Update HUD Wrapper State & Ambient Glow
  if (hudScoreWrapper) {
    hudScoreWrapper.className = `hud-score-wrapper ${isReliable ? 'is-reliable' : 'is-misleading'}`;
  }

  // 2. Animate Score Number
  animateScoreCounter(mainScore);

  // 3. Update Status Icon & Verdict Badges
  if (hudStatusIcon) {
    hudStatusIcon.textContent = isReliable ? '🛡️' : '⚠️';
  }

  if (scoreLabelText) {
    scoreLabelText.textContent = isReliable ? 'AUTHENTIC / VERIFIED' : 'MISLEADING / HIGH RISK';
  }

  if (hudSummaryBadge) {
    hudSummaryBadge.textContent = isReliable ? 'VERIFIED CREDIBILITY' : 'ANOMALY DETECTED';
  }

  if (hudVerdictPill) {
    hudVerdictPill.textContent = isReliable ? '✓ Reliable Content' : '⚠️ Misleading Content';
    hudVerdictPill.className = `hud-verdict-pill ${isReliable ? 'pill-reliable' : 'pill-misleading'}`;
  }

  if (hudConfidenceTag) {
    hudConfidenceTag.textContent = `${result.confidence}% Verdict Confidence`;
  }

  // 4. Update Detailed Breakdown Bars
  if (reliablePatternBar) reliablePatternBar.style.width = `${reliableScore}%`;
  if (misleadingPatternBar) misleadingPatternBar.style.width = `${misleadingScore}%`;
  if (reliablePatternValue) reliablePatternValue.textContent = `${reliableScore.toFixed(1)}%`;
  if (misleadingPatternValue) misleadingPatternValue.textContent = `${misleadingScore.toFixed(1)}%`;

  if (reliableSubtext) {
    reliableSubtext.textContent = isReliable
      ? 'High linguistic alignment with verified information.'
      : 'Low signal support for factual reliability.';
  }

  if (misleadingSubtext) {
    misleadingSubtext.textContent = isReliable
      ? 'Minimal deceptive or clickbait risk identified.'
      : 'High frequency of deceptive, clickbait, or scam patterns.';
  }

  // 5. Update Signal Cards List
  if (signalList) {
    signalList.innerHTML = (result.signals || [])
      .map(
        (sig) => `
        <div class="signal-card signal-${sig.severity}">
          <div class="signal-card-header">
            <span class="signal-title">${escapeHtml(sig.name)}</span>
            <span class="signal-badge status-${sig.severity}">${escapeHtml(sig.status)}</span>
          </div>
          <p class="signal-detail">${escapeHtml(sig.detail)}</p>
        </div>
      `
      )
      .join('');
  }
}

// ==========================================
// Copy Current Analysis Report
// ==========================================
async function copyCurrentAnalysis() {
  if (!currentAnalysisData) return;

  const isRel = currentAnalysisData.label === 'Reliable';
  const report = [
    `AUTHENTIQ AI VERDICT REPORT`,
    `===========================`,
    `Status: ${currentAnalysisData.label}`,
    `Confidence: ${currentAnalysisData.confidence}%`,
    `Reliability Score: ${currentAnalysisData.reliable_score}%`,
    `Misleading Risk: ${currentAnalysisData.misleading_score}%`,
    ``,
    `Article Excerpt:`,
    articleInput.value.slice(0, 300) + (articleInput.value.length > 300 ? '...' : ''),
  ].join('\n');

  try {
    await navigator.clipboard.writeText(report);
    if (copyResultBtn) {
      const origText = copyResultBtn.innerHTML;
      copyResultBtn.innerHTML = '✓ Copied!';
      setTimeout(() => {
        copyResultBtn.innerHTML = origText;
      }, 1800);
    }
    showNotification('Analysis report copied to clipboard!', 'success');
  } catch (err) {
    showNotification('Could not copy to clipboard.', 'error');
  }
}

// ==========================================
// Load & Render Isolated Local Browser History
// ==========================================
function loadHistory() {
  if (!historyList) return;

  const items = getLocalHistory();

  if (historyCount) historyCount.textContent = items.length;

  if (items.length === 0) {
    historyList.innerHTML = '<div class="empty-state">No saved analyses yet. Run an analysis to see records here.</div>';
    return;
  }

  historyList.innerHTML = items
    .map((item) => {
      const isRel = item.label === 'Reliable';
      const dateStr = item.created_at ? new Date(item.created_at).toLocaleString() : 'Recent';

      return `
        <div class="history-entry ${isRel ? 'history-reliable' : 'history-misleading'}">
          <div class="history-entry-head">
            <div>
              <span class="history-badge ${isRel ? 'badge-reliable' : 'badge-misleading'}">
                ${isRel ? '✓ Reliable' : '⚠️ Misleading'}
              </span>
              <span class="history-date">${escapeHtml(dateStr)}</span>
            </div>
            <strong class="history-conf">${Number(item.confidence).toFixed(1)}% Conf.</strong>
          </div>
          <p class="history-preview">${escapeHtml(item.article)}</p>
          <div class="history-actions-inline">
            <button class="history-action view-btn" data-id="${item.id}">🔍 View</button>
            <button class="history-action export-btn" data-export-id="${item.id}">📥 Export TXT</button>
            <button class="history-action delete-btn" data-delete-id="${item.id}">🗑️ Delete</button>
          </div>
        </div>
      `;
    })
    .join('');

  // Attach Action Listeners for Local Items
  historyList.querySelectorAll('.view-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-id');
      const items = getLocalHistory();
      const item = items.find((x) => x.id === id);
      if (!item) return;

      articleInput.value = item.article;
      updateTextStats();
      currentAnalysisData = {
        label: item.label,
        confidence: item.confidence,
        reliability: item.reliability,
        reliable_score: Number((item.reliability * 100).toFixed(1)),
        misleading_score: Number(((1.0 - item.reliability) * 100).toFixed(1)),
        signals: item.signals,
        metrics: item.metrics,
      };
      updatePredictionDisplay(currentAnalysisData);
      setPanel('analyzer');
      showNotification('Loaded historic analysis record.', 'success');
    });
  });

  historyList.querySelectorAll('.delete-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-delete-id');
      let items = getLocalHistory();
      items = items.filter((x) => x.id !== id);
      saveLocalHistory(items);
      loadHistory();
      showNotification('Record deleted.', 'success');
    });
  });

  historyList.querySelectorAll('.export-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-export-id');
      const items = getLocalHistory();
      const item = items.find((x) => x.id === id);
      if (!item) return;

      const signalsTxt = (item.signals || [])
        .map((s) => `• ${s.name} [${s.status}]: ${s.detail}`)
        .join('\n');

      const content = [
        `AUTHENTIQ AI ANALYSIS REPORT`,
        `===========================`,
        `Date: ${new Date(item.created_at).toLocaleString()}`,
        `Verdict: ${item.label}`,
        `Confidence: ${item.confidence}%`,
        `Reliability Score: ${Number(item.reliability * 100).toFixed(1)}%`,
        ``,
        `DETECTED SIGNALS:`,
        signalsTxt,
        ``,
        `ARTICLE CONTENT:`,
        item.article,
      ].join('\n');

      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `authentiq-analysis-${id}.txt`;
      a.click();
      URL.revokeObjectURL(url);
      showNotification('Report file exported.', 'success');
    });
  });
}

// ==========================================
// Clear All Local History
// ==========================================
function clearAllHistory() {
  if (!confirm('Are you sure you want to delete all saved analyses from your browser?')) return;

  localStorage.removeItem(LOCAL_STORAGE_HISTORY_KEY);
  loadHistory();
  showNotification('Browser history cleared successfully.', 'success');
}

// ==========================================
// Toast Notification Utility
// ==========================================
function showNotification(message, type = 'success') {
  const existing = document.querySelectorAll('.notification');
  existing.forEach((el) => el.remove());

  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.innerHTML = `${type === 'error' ? '⚠️' : '✓'} <span>${escapeHtml(message)}</span>`;

  document.body.appendChild(notification);

  setTimeout(() => {
    notification.classList.add('show');
  }, 10);

  setTimeout(() => {
    notification.classList.remove('show');
    setTimeout(() => notification.remove(), 300);
  }, 3500);
}

// ==========================================
// HTML Escape Helper
// ==========================================
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
