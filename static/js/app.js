/**
 * app.js — Main SPA logic for Predictive Maintenance Platform
 * Manages: routing, run buttons, pipeline state, code viewer, toasts
 */

/* ════════════════════════════════════════════════════════════
   CODE SNIPPETS (embedded Python for the Code Viewer)
════════════════════════════════════════════════════════════ */
const CODE = {
  load: `def load_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    local_path = "ai4i2020.csv"
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        df = pd.read_csv(url)
        df.to_csv(local_path, index=False)
    return df

df = load_dataset()
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print("Target Distribution (%):")
print((df["Machine failure"].value_counts(normalize=True) * 100).round(2))`,

  preprocess: `# Remove identifiers and leakage columns
leakage_and_ids = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
# Reasoning: TWF/HDF/PWF/OSF/RNF are sub-failure flags whose logical OR
# equals Machine failure exactly — keeping them causes 100% target leakage.
df_clean = df.drop(columns=leakage_and_ids).copy()

# Domain-driven feature engineering
# 1. Heat dissipation margin
df_clean["Temp_Diff_K"] = (
    df_clean["Process temperature [K]"] - df_clean["Air temperature [K]"]
)
# 2. True mechanical power P = τω  (Torque × angular velocity)
df_clean["Power_W"] = df_clean["Torque [Nm]"] * (
    df_clean["Rotational speed [rpm]"] * 2 * np.pi / 60.0
)
# 3. Cumulative tool strain
df_clean["Overstrain_Index"] = df_clean["Tool wear [min]"] * df_clean["Torque [Nm]"]

print(df_clean.describe().round(2).T[["min", "mean", "50%", "max", "std"]])`,

  eda: `import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
sns.countplot(data=df_clean, x="Machine failure", ax=axes[0],
              palette=["#A3A3A3", "#FF4500"])
sns.boxplot(data=df_clean, x="Machine failure", y="Power_W", ax=axes[1],
            palette=["#A3A3A3", "#FF4500"])
sns.boxplot(data=df_clean, x="Machine failure", y="Overstrain_Index", ax=axes[2],
            palette=["#A3A3A3", "#FF4500"])
plt.tight_layout()
plt.savefig("fig1_eda_distributions.png", dpi=150)

# Correlation heatmap
plt.figure(figsize=(9, 7))
numeric_df = df_clean.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.savefig("fig2_correlation_heatmap.png", dpi=150)`,

  split: `X = df_clean.drop(columns=["Machine failure"])
y = df_clean["Machine failure"]

# Hold out 15% untouched Test Set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
)
# Split remaining 85% → 70% Train + 15% Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.17647, stratify=y_temp, random_state=RANDOM_STATE
)

# Preprocessing pipeline (fitted ONLY on X_train → no leakage)
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features  = ["Type"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_features),
])

print(f"Train : {len(X_train)} rows | Failures: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Val   : {len(X_val)} rows | Failures: {y_val.sum()} ({y_val.mean()*100:.2f}%)")
print(f"Test  : {len(X_test)} rows | Failures: {y_test.sum()} ({y_test.mean()*100:.2f}%)")`,

  train: `models = {
    "Baseline: Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    "Approach 1: Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
    "Approach 2: Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
val_results = []

for name, estimator in models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])

    # 5-fold cross-validation on TRAINING set
    cv_f1 = cross_val_score(pipe, X_train, y_train,
                            cv=cv, scoring="f1", n_jobs=-1).mean()
    pipe.fit(X_train, y_train)

    val_pred = pipe.predict(X_val)
    val_prob = pipe.predict_proba(X_val)[:, 1]
    p, r, _  = precision_recall_curve(y_val, val_prob)

    val_results.append({
        "Model":         name,
        "5-Fold CV F1":  round(cv_f1, 4),
        "Val Precision": round(precision_score(y_val, val_pred), 4),
        "Val Recall":    round(recall_score(y_val,    val_pred), 4),
        "Val F1":        round(f1_score(y_val,        val_pred), 4),
        "Val PR-AUC":    round(auc(r, p), 4),
    })

print(pd.DataFrame(val_results).to_string(index=False))`,

  tune: `gb_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
])

param_grid = {
    "clf__n_estimators":  [150, 250],
    "clf__max_depth":     [3, 4],
    "clf__learning_rate": [0.05, 0.1],
}

grid_search = GridSearchCV(gb_pipe, param_grid,
                           cv=cv, scoring="f1", n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"Best Params : {grid_search.best_params_}")
print(f"Best CV F1  : {grid_search.best_score_:.4f}")`,

  evaluate: `test_pred = best_model.predict(X_test)
test_prob = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, test_pred, digits=4))

# Precision-Recall curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name, pipe in fitted_pipelines.items():
    probs    = pipe.predict_proba(X_test)[:, 1]
    p, r, _  = precision_recall_curve(y_test, probs)
    axes[0].plot(r, p, label=f"{name} (PR-AUC={auc(r,p):.3f})")

p_t, r_t, _ = precision_recall_curve(y_test, test_prob)
axes[0].plot(r_t, p_t, "--", lw=2.2, color="black",
             label=f"Tuned GB (PR-AUC={auc(r_t,p_t):.3f})")
axes[0].set_title("Precision-Recall Curve (Test Set)")

# Confusion matrix
cm   = confusion_matrix(y_test, test_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy","Failure"])
disp.plot(ax=axes[1], cmap="Blues", colorbar=False)
plt.tight_layout()
plt.savefig("fig3_model_evaluation.png", dpi=150)`,

  features: `ohe_names = best_model.named_steps["prep"]\
    .named_transformers_["cat"]\
    .get_feature_names_out(cat_features).tolist()
all_feature_names = num_features + ohe_names
importances       = best_model.named_steps["clf"].feature_importances_

feat_imp = pd.Series(importances, index=all_feature_names)\
             .sort_values(ascending=True)

plt.figure(figsize=(9, 5))
feat_imp.plot(kind="barh", color="#FF4500")
plt.title("Feature Importances (Tuned Gradient Boosting)")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig("fig4_feature_importance.png", dpi=150)`,

  errors: `audit_df = df.loc[X_test.index].copy()
audit_df["Predicted"]    = test_pred
audit_df["Failure_Prob"] = test_prob.round(3)

false_negatives = audit_df[
    (audit_df["Machine failure"] == 1) & (audit_df["Predicted"] == 0)
]
false_positives = audit_df[
    (audit_df["Machine failure"] == 0) & (audit_df["Predicted"] == 1)
]

print(f"Total Test Failures : {y_test.sum()}")
print(f"Caught (True Pos)   : {((y_test==1)&(test_pred==1)).sum()}")
print(f"Missed (False Neg)  : {len(false_negatives)}")
print(f"False Alarms (FP)   : {len(false_positives)}")
print("\\nFailure Modes in Missed Cases:")
print(false_negatives[["TWF","HDF","PWF","OSF","RNF"]].sum())`,
};

const CODE_FILENAMES = {
  load: 'step1_load_dataset.py',
  preprocess: 'step2_preprocess.py',
  eda: 'step3_eda.py',
  split: 'step4_split.py',
  train: 'step5_train_models.py',
  tune: 'step6_hyperparameter_tuning.py',
  evaluate: 'step7_evaluation.py',
  features: 'step8_feature_importance.py',
  errors: 'step9_error_analysis.py',
};

/* ════════════════════════════════════════════════════════════
   PIPELINE SECTION DEFINITIONS
════════════════════════════════════════════════════════════ */
const STEPS = [
  { id: 'load',       label: 'Data Loading',     endpoint: '/api/run/load',       render: 'renderLoad',       requires: [] },
  { id: 'preprocess', label: 'Preprocessing',    endpoint: '/api/run/preprocess', render: 'renderPreprocess', requires: ['load'] },
  { id: 'eda',        label: 'EDA',              endpoint: '/api/run/eda',        render: 'renderEDA',        requires: ['preprocess'] },
  { id: 'split',      label: 'Data Split',       endpoint: '/api/run/split',      render: 'renderSplit',      requires: ['preprocess'] },
  { id: 'train',      label: 'Model Training',   endpoint: '/api/run/train',      render: 'renderTrain',      requires: ['split'], slow: true },
  { id: 'tune',       label: 'Hyper-Tuning',     endpoint: '/api/run/tune',       render: 'renderTune',       requires: ['train'], slow: true },
  { id: 'evaluate',   label: 'Evaluation',       endpoint: '/api/run/evaluate',   render: 'renderEvaluate',   requires: ['tune'] },
  { id: 'features',   label: 'Feature Importance',endpoint: '/api/run/features',  render: 'renderFeatures',   requires: ['tune'] },
  { id: 'errors',     label: 'Error Analysis',   endpoint: '/api/run/errors',     render: 'renderErrors',     requires: ['evaluate'] },
];

const pipelineState = {};  // { stepId: { done, data } }

/* ════════════════════════════════════════════════════════════
   TOAST NOTIFICATIONS
════════════════════════════════════════════════════════════ */
function showToast(message, type = 'info', duration = 4000) {
  const icons = { success: 'bx-check-circle', error: 'bx-x-circle', warning: 'bx-error', info: 'bx-info-circle' };
  const tc = document.getElementById('toast-container');
  const t  = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<i class="bx ${icons[type]||icons.info}"></i><span>${message}</span>`;
  tc.appendChild(t);
  setTimeout(() => {
    t.style.animation = 'toastOut 0.3s ease forwards';
    setTimeout(() => t.remove(), 300);
  }, duration);
}

/* ════════════════════════════════════════════════════════════
   STATUS MANAGEMENT
════════════════════════════════════════════════════════════ */
function setStepStatus(stepId, status) {
  const badge = document.getElementById(`status-${stepId}`);
  const dot   = document.getElementById(`dot-${stepId}`);
  const btn   = document.getElementById(`run-btn-${stepId}`);
  const pf    = document.querySelector(`.pf-step[data-step="${stepId}"]`);

  if (!badge) return;

  badge.className = `sec-status-badge ${status}`;
  if (dot) {
    dot.className = `nav-dot ${status === 'done' ? 'done' : status === 'running' ? 'running' : status === 'error' ? 'error' : ''}`;
  }
  if (pf) {
    pf.className = `pf-step ${status}`;
  }

  const labels = { running: '⟳ Running…', done: '✓ Complete', error: '✕ Error' };
  badge.querySelector('span:last-child').textContent = labels[status] || 'Not Run';

  if (btn) {
    btn.className = `run-btn ${status === 'running' ? 'running' : status === 'done' ? 'done' : ''}`;
    btn.disabled  = status === 'running';
    if (status === 'running') {
      btn.innerHTML = '<i class="bx bx-loader-alt bx-spin"></i> Running…';
    } else if (status === 'done') {
      btn.innerHTML = '<i class="bx bx-check"></i> Re-run';
    } else {
      btn.innerHTML = '<i class="bx bx-play"></i> Run';
    }
  }
}

function updateProgress() {
  const done = STEPS.filter(s => pipelineState[s.id]?.done).length;
  const frac = document.getElementById('progress-fraction');
  const bar  = document.getElementById('progress-bar');
  if (frac) frac.textContent = `${done} / ${STEPS.length}`;
  if (bar)  bar.style.width  = `${(done / STEPS.length) * 100}%`;
}

/* ════════════════════════════════════════════════════════════
   LOADING OVERLAY
════════════════════════════════════════════════════════════ */
let _timerInterval = null;
function showLoading(title, sub) {
  const ov = document.getElementById('loading-overlay');
  document.getElementById('loading-title').textContent = title;
  document.getElementById('loading-sub').textContent   = sub;
  document.getElementById('loading-timer').textContent = '0s';
  ov.classList.remove('hidden');
  let sec = 0;
  _timerInterval = setInterval(() => {
    document.getElementById('loading-timer').textContent = `${++sec}s`;
  }, 1000);
}
function hideLoading() {
  clearInterval(_timerInterval);
  document.getElementById('loading-overlay').classList.add('hidden');
}

/* ════════════════════════════════════════════════════════════
   RUN A SINGLE STEP
════════════════════════════════════════════════════════════ */
async function runStep(stepId, silent = false) {
  const step = STEPS.find(s => s.id === stepId);
  if (!step) return;

  // Check prerequisites
  for (const req of step.requires) {
    if (!pipelineState[req]?.done) {
      showToast(`⚠ Please run "${req}" step first`, 'warning');
      return false;
    }
  }

  setStepStatus(stepId, 'running');
  if (!silent && step.slow) showLoading(`Running ${step.label}…`, 'This may take 1–2 minutes…');

  try {
    const data = await API.post(step.endpoint);
    pipelineState[stepId] = { done: true, data };

    // Render results
    Charts[step.render](data, `results-${stepId}`);

    setStepStatus(stepId, 'done');
    updateProgress();
    if (!silent) {
      showToast(`✓ ${step.label} complete!`, 'success');
      // Smooth-scroll results into view
      const resEl = document.getElementById(`results-${stepId}`);
      if (resEl) setTimeout(() => resEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 300);
    }
    return true;
  } catch (err) {
    setStepStatus(stepId, 'error');
    if (!silent) showToast(`Error in ${step.label}: ${err.message}`, 'error', 7000);
    console.error(`Step ${stepId}:`, err);
    return false;
  } finally {
    if (!silent && step.slow) hideLoading();
  }
}

/* ════════════════════════════════════════════════════════════
   RUN ALL PIPELINE
════════════════════════════════════════════════════════════ */
async function runAllPipeline() {
  const btn = document.getElementById('run-all-btn');
  btn.disabled = true;
  btn.innerHTML = '<i class="bx bx-loader-alt bx-spin"></i> Running…';
  showLoading('Running Full Pipeline…', 'Loading → Preprocessing → EDA → Split → Training → Tuning → Evaluation (≈ 3–5 min)');

  for (const step of STEPS) {
    document.getElementById('loading-title').textContent = `Running: ${step.label}…`;
    const ok = await runStep(step.id, true);
    if (!ok) break;
  }

  hideLoading();
  btn.disabled = false;
  btn.innerHTML = '<i class="bx bx-play-circle"></i> Run Full Pipeline';

  const done = STEPS.filter(s => pipelineState[s.id]?.done).length;
  if (done === STEPS.length) {
    showToast('🎉 Full pipeline complete! All results loaded.', 'success', 6000);
  } else {
    showToast(`Pipeline stopped at step ${done + 1}. Check errors above.`, 'error', 6000);
  }
}

/* ════════════════════════════════════════════════════════════
   CODE VIEWER
════════════════════════════════════════════════════════════ */
let _currentCodeTab = 'load';

function initCodeViewer() {
  const display  = document.getElementById('cv-code-display');
  const filename = document.getElementById('cv-filename');

  function switchTab(key) {
    _currentCodeTab = key;
    display.textContent  = CODE[key] || '';
    filename.textContent = CODE_FILENAMES[key] || `${key}.py`;
    Prism.highlightElement(display);
    document.querySelectorAll('.cvt-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === `cv-${key}`);
    });
  }

  document.querySelectorAll('.cvt-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.tab.replace('cv-', '');
      switchTab(key);
    });
  });

  document.getElementById('cv-copy-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(CODE[_currentCodeTab] || '').then(() => {
      showToast('Code copied to clipboard!', 'success', 2000);
    });
  });

  // Initialise with 'load'
  switchTab('load');
}

/* ════════════════════════════════════════════════════════════
   INLINE CODE SNIPPETS (per section)
════════════════════════════════════════════════════════════ */
function injectSectionCode() {
  Object.keys(CODE).forEach(key => {
    const el = document.getElementById(`code-txt-${key}`);
    if (el) {
      el.textContent = CODE[key];
      Prism.highlightElement(el);
    }
  });
}

function initCodeToggles() {
  document.querySelectorAll('.code-toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key   = btn.dataset.code;
      const wrap  = document.getElementById(`code-${key}`);
      if (!wrap) return;
      const isOpen = !wrap.classList.contains('hidden');
      wrap.classList.toggle('hidden', isOpen);
      btn.innerHTML = isOpen
        ? '<i class="bx bx-code-alt"></i> View Code'
        : '<i class="bx bx-code-alt"></i> Hide Code';
    });
  });
}

/* ════════════════════════════════════════════════════════════
   SIDEBAR NAVIGATION + ACTIVE STATE
════════════════════════════════════════════════════════════ */
function initNavigation() {
  const navItems = document.querySelectorAll('.nav-item[data-target]');

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id.replace('sec-', '');
        navItems.forEach(a => {
          a.classList.toggle('active', a.dataset.target === id);
        });
      }
    });
  }, { threshold: 0.25 });

  document.querySelectorAll('.sec[id]').forEach(sec => observer.observe(sec));
}

/* ════════════════════════════════════════════════════════════
   SERVER STATUS CHECK
════════════════════════════════════════════════════════════ */
async function checkServer() {
  const el = document.getElementById('server-status');
  try {
    const res = await API.get('/api/status');
    el.className = 'server-status online';
    el.innerHTML = '<span class="status-dot-sm"></span> API Connected';

    // If backend has pre-computed steps (e.g., after a reload), reflect them
    if (res.pipeline_status) {
      const st = res.pipeline_status;
      const stepMap = {
        loaded: 'load', preprocessed: 'preprocess', eda: 'eda',
        split: 'split', trained: 'train', tuned: 'tune',
        evaluated: 'evaluate', features: 'features', errors: 'errors',
      };
      Object.entries(stepMap).forEach(([k, stepId]) => {
        if (st[k]) {
          pipelineState[stepId] = { done: true, data: null };
          setStepStatus(stepId, 'done');
        }
      });
      updateProgress();
    }
  } catch {
    el.className = 'server-status offline';
    el.innerHTML = '<span class="status-dot-sm"></span> API Offline';
    showToast('⚠ Cannot reach Flask API. Run: python app.py', 'error', 8000);
  }
}

/* ════════════════════════════════════════════════════════════
   RUN BUTTON WIRE-UP
════════════════════════════════════════════════════════════ */
function initRunButtons() {
  document.querySelectorAll('.run-btn[data-section]').forEach(btn => {
    btn.addEventListener('click', () => {
      const stepId = btn.dataset.section;
      const step   = STEPS.find(s => s.id === stepId);
      if (step?.slow) {
        showLoading(`Running ${step.label}…`, 'This may take 1–2 minutes…');
      }
      runStep(stepId).finally(() => {
        if (step?.slow) hideLoading();
      });
    });
  });

  document.getElementById('run-all-btn').addEventListener('click', runAllPipeline);
}

/* ════════════════════════════════════════════════════════════
   BOOT
════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  checkServer();
  initNavigation();
  initRunButtons();
  initCodeViewer();
  injectSectionCode();
  initCodeToggles();
  updateProgress();
});
