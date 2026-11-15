# ⚔️ Jon Snow: The Data Preprocessing Quest

**An Interactive 2D Action-Strategy Browser Game for Learning Tabular Data Preprocessing**  
*Introduction to Data Science & Engineering (IDSE) — Interactive Curriculum Project*

### 👥 Team Members
* **Bikram Bose** — Roll No: `EC24B1049`
* **Vishnesh** — Roll No: `EC24B1086`

---

## 🚀 How to Open & Run the Game from GitHub

### Option 1: Play Directly in Browser via GitHub Pages (Recommended)
1. Upload `jon_snow_the_data_preprocessing_quest.html`, `back.png`, and `snow1.png` into the root folder of your GitHub repository (or rename the HTML file to `index.html` for a cleaner URL).
2. In your GitHub repository, click **Settings** $\rightarrow$ **Pages** (in the left sidebar).
3. Under **Build and deployment** $\rightarrow$ **Source**, select **Deploy from a branch**.
4. Choose `main` (or `master`) and `/ (root)`, then click **Save**.
5. Wait 1–2 minutes and open your live link:
   * If kept as original filename: `https://<your-github-username>.github.io/<repo-name>/jon_snow_the_data_preprocessing_quest.html`
   * If renamed to `index.html`: `https://<your-github-username>.github.io/<repo-name>/`

### Option 2: Clone or Download and Run Locally
1. **Clone the repository** (or click **Code $\rightarrow$ Download ZIP** and extract it):
   ```bash
   git clone https://github.com/<your-github-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Ensure `jon_snow_the_data_preprocessing_quest.html`, `back.png`, and `snow1.png` are in the **same folder**.
3. Double-click `jon_snow_the_data_preprocessing_quest.html` to open it in any modern browser (**Chrome**, **Edge**, **Firefox**, or **Safari**).
4. Click **START QUEST (LEVEL 1)** — no server, Python environment, or package installation is required.

---

## 📖 1. Overview

**Jon Snow: The Data Preprocessing Quest** is a 2D action-strategy browser game designed to teach foundational data preprocessing concepts to students with little or no prior hands-on experience in data science. Set in a Westeros-inspired battlefield, players guide Jon Snow across four progressive stages where enemy bosses are protected by impenetrable data shields. Weapons alone cannot harm the villains; instead, players must gather corrupted battlefield intelligence scrolls and clean, transform, and validate real tabular datasets inside an interactive **Data Workbench** to shatter each boss's shield.

* **Core Concept:** End-to-end tabular data preprocessing covering schema inspection, data types, missing value detection and median imputation, record deduplication, string standardization, Interquartile Range (IQR) outlier filtering, categorical label encoding, Min-Max feature scaling, and pipeline ordering.
* **Target Audience:** Undergraduate beginners and first-time data science learners. No prior coding syntax is required to play, while a live `pandas` / `scikit-learn` execution log displays the exact Python equivalents of every in-game operation in real time.
* **Structure & Length:** 4 structured campaign levels (spanning Jon Snow's progression from Age 19 to Age 23), playable in approximately 10–15 minutes.
* **Format & Deployment:** Self-contained single-file web application (`jon_snow_the_data_preprocessing_quest.html`) built with HTML5 Canvas, vanilla JavaScript, and Tailwind CSS. Runs offline in any modern desktop or mobile browser with zero installation.

---

## 🎯 2. Learning Objectives

Upon completing the four campaign levels, a learner is able to perform and explain the following core preprocessing tasks, each reinforced directly through gameplay constraints and visual table updates:

1. **Raw Data & Schema Inspection (Level 1 — Age 19):** Inspect a tabular DataFrame (`df.dtypes`, `df.isnull()`), distinguish numerical columns (`Soldier_ID`, `Age`, `Strength`) from categorical text columns (`Weapon`), and identify incomplete records containing `NaN` values.
2. **Data Cleaning & Imputation (Level 2 — Age 21):** Repair missing numerical entries using median imputation (`fillna(median)`), eliminate duplicate rows (`drop_duplicates`), and standardize inconsistent string casing (`str.title()`) so categorical groupings are unified.
3. **Outlier Filtering, Encoding & Normalization (Level 3 — Age 22):** Detect extreme anomalies using the IQR fence rule ($Q_3 + 1.5 \times \text{IQR}$), convert categorical text labels into integer representations via `LabelEncoder`, and rescale numerical features into $[0, 1]$ via Min-Max Normalization—while understanding why outliers must be removed *before* scaling.
4. **End-to-End Preprocessing Pipeline Mastery (Level 4 — Age 23):** Execute a complete, sequential preprocessing pipeline (`Clean` $\rightarrow$ `Filter Outliers` $\rightarrow$ `Encode & Scale`) on an unseen, multi-anomaly dataset under active combat pressure, enforcing strict operation order.

---

## 🎮 3. How to Play

### Flow of a Run
1. **Explore & Gather Intelligence:** Move Jon Snow across the 2D battlefield arena while dodging patrolling hazards (ice spikes, wights, wildfire, and dragon fireballs) and press `SPACE` near the 3 glowing data scrolls to load the level's dataset.
2. **Open the Data Workbench:** Press `TAB` or `E` (or click the **Data Workbench** button) to inspect the live HTML DataFrame table, row/column dimensions, missing value count, and duplicate count.
3. **Apply Preprocessing Operations:** Press keys `1`, `2`, or `3` (or click the operation cards) to clean, filter, encode, and normalize the table while observing live changes in the table cells, distribution bar chart, and Python backend log.
4. **Validate & Shatter the Shield:** Press `ENTER` to validate the DataFrame. If all required preprocessing steps are complete, the boss's protective shield shatters; if steps are missing or executed out of order, Jon loses HP with an explanatory diagnostic message.
5. **Defeat the Boss & Advance:** Approach the unshielded boss on the arena and strike twice with Longclaw (`SPACE`) to clear the stage and unlock the next chapter.

### Controls & HUD Mechanics

| Action | Keyboard Shortcut | On-Screen / Gameplay Effect |
| :--- | :--- | :--- |
| **Move Jon Snow** | `Arrow Keys` or `W A S D` | Navigates the $800 \times 500$ arena and dodges moving hazards. |
| **Collect Scroll / Strike** | `SPACE` | Collects nearby data scrolls (+50 pts) or swings Longclaw at the boss. |
| **Toggle Data Workbench** | `TAB` or `E` | Opens/closes the live DataFrame inspection and transformation modal. |
| **Execute Pipeline Step** | `1`, `2`, `3` | Runs the corresponding `pandas` / `sklearn` operation in the Workbench. |
| **Validate DataFrame** | `ENTER` | Verifies data cleanliness and shatters the boss's shield (+250 pts). |
| **Toggle Learning Hint** | `H` | Displays or hides the contextual step-by-step preprocessing hint banner. |

### Arena Entities & Scoring

| Entity / Indicator | Visual Representation | Gameplay & Educational Role |
| :--- | :--- | :--- |
| **Data Scrolls (#1–#3)** | Pulsing amber scroll icons | Represent distributed batches of raw battlefield records; all 3 must be gathered before opening the Workbench. |
| **Boss Shield Aura** | Dashed cyan ring $\rightarrow$ Solid green ring | Blocks all sword damage (-15 HP recoil if attacked early) until the dataset passes validation. |
| **Arena Hazards** | Ice spikes, wights, wildfire, dragonfire | Patrolling projectiles that inflict -10 HP on contact, encouraging spatial awareness during data gathering. |
| **Scoring & Health** | 100 Jon HP / 100 Boss HP / Score HUD | +50 pts per scroll, +250 pts per validated dataset, +150 pts per sword hit; wrong pipeline order deducts -10 HP. |

---

## 🧩 4. Game Design and Learning Loop

Every stage embeds specific data anomalies inside a structured DataFrame so the learner experiences *why* each preprocessing step is necessary before moving to the next stage.

| Level & Boss | Core Concept | Dataset on the Board | What the Player Discovers | Player's Decision |
| :--- | :--- | :--- | :--- | :--- |
| **1. Wildling King** *(Age 19)* | Raw Data & Schema Inspection | 4 rows $\times$ 4 cols (`id`, `age`, `weapon`, `strength`) with 1 `NaN` | Columns have distinct numerical vs categorical types; missing values break model readiness. | Classify column `dtypes` and flag corrupt row (ID 102). |
| **2. The Night King** *(Age 21)* | Data Cleaning (NaNs, Dups, Case) | 5 rows with missing Age, duplicate ID 103, and mixed-case weapons (`'bow'`, `'SWORD'`) | Duplicate IDs inflate counts and inconsistent casing fragments categories. | Impute median Age, drop duplicate ID 103, standardize Title Case. |
| **3. Queen Cersei** *(Age 22)* | IQR Outliers, Encoding & Scaling | 5 rows including extreme outlier ID 203 (`Age=150`, `Strength=200`) | Scaling before removing outliers squashes valid data near `0.0`; models require numeric codes. | Remove IQR outlier first, label-encode weapons, Min-Max scale to $[0, 1]$. |
| **4. Daenerys** *(Age 23 — Final)* | End-to-End Pipeline Mastery | 6 unseen rows combining `NaN`, duplicate ID 303, outlier ID 304 (`Age=165`), and raw text | Preprocessing is order-dependent: cleaning must precede outlier removal, which must precede scaling. | Execute Stage 1 (Clean) $\rightarrow$ Stage 2 (Outlier) $\rightarrow$ Stage 3 (Encode/Scale) in order. |

### The 6-Step Pedagogical Loop in Every Level
1. **Brief & Mission Checklist:** The right-hand sidebar presents the narrative context, a 4-step interactive checklist, and the initial Python `pandas` / `scikit-learn` script.
2. **Gather & Dodge:** Players actively collect all 3 scrolls on the canvas, linking data acquisition with spatial gameplay.
3. **Inspect Live Metrics:** Opening the Data Workbench immediately computes and displays table shape (`Rows × Columns`), missing value count, duplicate count, and dataset readiness status.
4. **Transform with Instant Visual & Code Feedback:** Each preprocessing operation updates the HTML table cells in place, animates the feature distribution bars (Levels 3–4), and appends the corresponding Python snippet to the live execution log.
5. **Order-Aware Guardrails:** Attempting to normalize before removing outliers in Level 3—or skipping prerequisite stages in Level 4—triggers a `-10 HP` penalty and explains the exact mathematical consequence.
6. **Validate & Combat Resolution:** Passing validation shatters the boss's shield, allowing Jon Snow to defeat the boss and advance to the next stage.

---

## ⚙️ 5. Technical Implementation & Mathematical Formulation

The application is architected as a single self-contained HTML file (`jon_snow_the_data_preprocessing_quest.html`) requiring no backend server or build step. The 2D battlefield renders at 60 FPS via `requestAnimationFrame` on an $800 \times 500$ HTML5 Canvas (supporting custom sprite assets `back.png` and `snow1.png` with automatic vector fallbacks), while the Data Workbench dynamically mutates in-memory JavaScript record arrays mirroring `pandas` DataFrames.

| Preprocessing Feature | Algorithmic Implementation in Game Engine |
| :--- | :--- |
| **Median Imputation** | Extracts non-null `Age` values, sorts ascending, and computes the middle value (or average of the two central elements) to fill `null` entries without distortion from skewed values. |
| **Deduplication & Case Normalization** | Filters records via a `Set` on `Soldier_ID` (`keep='first'`) and applies `charAt(0).toUpperCase() + slice(1).toLowerCase()` to unify categorical labels. |
| **IQR Outlier Filtering** | Evaluates quartile bounds ($\text{IQR} = Q_3 - Q_1$) and removes records exceeding the upper fence $Q_3 + 1.5 \times \text{IQR}$ (eliminating injected anomalies such as `Age = 150` and `Strength = 200`). |
| **Categorical Label Encoding** | Maps standardized string categories (`'Sword'`, `'Bow'`, `'Spear'`) to discrete integers (`0`, `1`, `2`) in a new `Weapon_Encoded` column. |
| **Min-Max Feature Normalization** | Rescales cleaned `Age` and `Strength` columns into $[0.00, 1.00]$ and updates the live horizontal distribution bars. |
| **Live Python Code Telemetry** | Appends formatted `pandas` and `sklearn.preprocessing` code blocks to the sidebar console after every player action. |

### Formulas Used
* **Min-Max Scaling:**
  $$X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$
* **IQR Upper Fence:**
  $$\text{Upper Fence} = Q_3 + 1.5 \times (Q_3 - Q_1)$$

---

## 📊 6. Evidence of Learning & Playtest Evaluation

To evaluate pedagogical effectiveness, a structured peer playtest was conducted across 5 undergraduate participants with varying programming and data science backgrounds. Each participant completed a 5-question conceptual diagnostic check before playing Level 1 and answered the same 5-question assessment after completing Level 4, alongside logging completion time, hardest stage, and common preprocessing mistakes.

| Tester | Background | Pre-Test (/5) | Post-Test (/5) | Play Time | Hardest Level | Key Observation & Feedback |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **P1** | Beginner in Python | 2 | 5 | 12 min | Level 3 (Outliers & Scaling) | Initially tried normalizing before dropping `Age=150`; HP penalty made the scaling distortion click immediately. |
| **P2** | Basic Data Science | 3 | 5 | 10 min | Level 4 (Full Pipeline) | Liked seeing the live `pandas` code log update side-by-side with the DataFrame table. |
| **P3** | First-Time Learner | 1 | 4 | 14 min | Level 2 (Data Cleaning) | Thought `NaN` could be ignored at first; appreciated the visual row highlighting and hint toggle. |
| **P4** | Intermediate Python | 4 | 5 | 9 min | Level 4 (Full Pipeline) | Great interactive revision tool; dodging dragon fireballs kept engagement high between workbench stages. |
| **P5** | Basic Python | 2 | 4 | 13 min | Level 3 (Outliers & Scaling) | Found the bar chart visualization of outlier compression very intuitive compared to textbook formulas. |

> **Playtest Summary:** Average Pre-Test Score = **2.4 / 5** $\rightarrow$ Average Post-Test Score = **4.6 / 5** (**+2.2** average gain; **5 of 5** testers improved).