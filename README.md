# Rajasthan Royals SupeRR Selector (Season 2) — Round 1

> **Data-Driven Identification & Ranking of the Most Reliable Players Under Pressure in T20 Cricket (IPL & T20I, 2022–Present)**

---

## 📌 Executive Summary

In high-stakes T20 cricket, standard aggregates (such as raw Strike Rate or Economy Rate) conflate low-leverage "garbage time" stat-padding with genuine high-pressure match execution. 

This repository presents the official **Rajasthan Royals SupeRR Selector Season 2 (Round 1)** analytics framework. It introduces a mathematically rigorous, delivery-level **Pressure Index ($PI \in [0, 1]$)** that isolates the top **24.1%** most critical match situations across all IPL and T20 International matches from 2022 onwards. It then quantifies individual player performance using contextual **Run Value Added (RVA)**, **Empirical Bayes shrinkage**, and **match-level consistency scoring**.

---

## 🏆 Official Top 10 Player Rankings (Unstop Selection)

### 🏏 Top 10 Batters Under Pressure
| Rank | Player ID | Batter | HP Balls | Ord Balls | HP Matches | HP Impact (RVA/100) | Ordinary Base | Clutch $\Delta$ | Consistency % | Final Rating |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 6396 | **Tim David** | 519 | 707 | 56 | +81.99 | +24.82 | +57.17 | 67.86% | **60.26** |
| **2** | 6726 | **Romario Shepherd** | 413 | 274 | 50 | +68.04 | -36.50 | +83.04 | 72.00% | **57.42** |
| **3** | 37052 | **Naman Dhir** | 212 | 303 | 21 | +66.64 | -12.41 | +79.06 | 66.67% | **48.86** |
| **4** | 4708 | **Rinku Singh** | 646 | 638 | 58 | +61.21 | +10.91 | +50.31 | 62.07% | **48.13** |
| **5** | 47 | **Virat Kohli** | 185 | 2271 | 18 | +69.50 | +20.67 | +48.83 | 66.67% | **43.37** |
| **6** | 4703 | **Rishabh Pant** | 329 | 910 | 35 | +51.95 | -5.61 | +57.55 | 68.57% | **41.97** |
| **7** | 5939 | **Cameron Green** | 230 | 842 | 20 | +59.84 | +22.51 | +37.33 | 80.00% | **40.25** |
| **8** | 20702 | **Tristan Stubbs** | 371 | 710 | 38 | +55.39 | +15.13 | +40.26 | 60.53% | **40.01** |
| **9** | 1318 | **Andre Russell** | 302 | 431 | 33 | +55.17 | +11.17 | +44.00 | 60.61% | **39.54** |
| **10** | 7897 | **Namboori Tilak Varma** | 633 | 1785 | 55 | +51.02 | +23.72 | +27.30 | 65.45% | **37.40** |

---

### 🎯 Top 10 Bowlers Under Pressure
| Rank | Player ID | Bowler | HP Balls | Ord Balls | HP Matches | HP Impact (RVS/24) | Ordinary Base | Clutch $\Delta$ | Consistency % | Final Rating |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 4900 | **Wanindu Hasaranga** | 353 | 1681 | 40 | +16.38 | +2.45 | +13.94 | 70.00% | **11.05** |
| **2** | 829 | **Mitchell Starc** | 225 | 578 | 27 | +14.71 | -2.46 | +17.16 | 51.85% | **9.20** |
| **3** | 8661 | **Matheesha Pathirana** | 444 | 688 | 36 | +12.10 | +2.12 | +9.98 | 72.22% | **8.59** |
| **4** | 781 | **Sunil Narine** | 241 | 1289 | 31 | +13.93 | +4.48 | +9.44 | 58.06% | **7.79** |
| **5** | 2382 | **Jamie Overton** | 177 | 320 | 21 | +13.40 | +1.54 | +11.86 | 66.67% | **7.32** |
| **6** | 3644 | **Kuldeep Yadav** | 402 | 1512 | 50 | +10.89 | +3.90 | +6.99 | 60.00% | **6.95** |
| **7** | 2933 | **Jasprit Bumrah** | 556 | 1152 | 54 | +10.54 | +4.78 | +5.76 | 46.30% | **6.74** |
| **8** | 4265 | **T Natarajan** | 423 | 629 | 37 | +5.86 | -7.66 | +13.52 | 51.35% | **5.80** |
| **9** | 7364 | **Haris Rauf** | 328 | 699 | 32 | +8.31 | +0.32 | +7.99 | 59.38% | **5.60** |
| **10** | 278 | **Tim Southee** | 196 | 630 | 25 | +8.93 | +2.23 | +6.70 | 52.00% | **4.74** |

---

## 🔬 Mathematical Methodology

### 1. Delivery-Level Pressure Index ($PI \in [0, 1]$)
The pressure experienced on each ball is computed strictly before the delivery is bowled:

- **Innings 2 (Chasing):**
  $$PI = 0.45 \cdot P_{\text{RRR}} + 0.30 \cdot P_{\text{Wickets}} + 0.25 \cdot P_{\text{Endgame}}$$
  - **Required Run Rate Pacing:** $P_{\text{RRR}} = \frac{1}{1 + e^{-3.2(RRR / ParRPO - 1.22)}}$
  - **Wicket Exposure Curve:** $P_{\text{Wickets}} = \left(\frac{W_{\text{lost}}}{10}\right)^{1.3}$
  - **Endgame Decay:** $P_{\text{Endgame}} = \text{clip}\left(\frac{60 - \text{BallsRemaining}}{60}, 0, 1\right)$
  - *Garbage-Time Dampener:* If $RRR > 24$ with $8+$ wickets down, or $RRR < 4.5$ with $\le 3$ down, $PI$ is forced to $0.25$.

- **Innings 1 (Target Setting):**
  - **Top-Order Collapse:** $\le 30\%$ balls & $\ge 2$ down $\implies PI = 0.75$; $\le 50\%$ balls & $\ge 3$ down $\implies PI = 0.72$.
  - **Death Overs:** $\ge 75\%$ balls completed $\implies PI = 0.70 + 0.03 \cdot W_{\text{lost}}$.
  - **Scoring Deficit:** Compares current run rate against competition par run rate.

- **High Pressure Threshold:** $PI \ge 0.65$.

### 2. Empirical Validation
| Situation | Deliveries | % Deliveries | Implied Run Rate (RPO) | Boundary % | Wicket Rate % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ordinary ($PI < 0.65$)** | 147,234 | 75.9% | 8.39 | 17.95% | 4.87% |
| **High Pressure ($PI \ge 0.65$)** | 46,754 | 24.1% | 8.90 | 18.43% | **7.06%** (+44.8%) |

Under high-pressure conditions, the wicket hazard rate surges by **+44.8%**, proving that $PI \ge 0.65$ captures high-stress situations.

### 3. Contextual Run Value & Shrinkage
- **Batting (Run Value Added per 100 Balls):**
  $$\text{RVA} = (\text{Runs} - \mathbb{E}[\text{Runs}]) - 12.0 \cdot (\text{Dismissal} - \mathbb{E}[\text{Dismissal}])$$
- **Bowling (Run Value Saved per 24 Balls / 4 Overs):**
  $$\text{RVS} = (\mathbb{E}[\text{Runs}] - \text{Runs}) + 11.0 \cdot (\text{Wickets} - \mathbb{E}[\text{Wickets}])$$
- **Bayesian Empirical Shrinkage:** Mitigates small sample variance using prior weight $K = 120$ balls.
- **Match Consistency Multiplier:** Adjusts rating by the fraction of high-pressure matches where the player contributed positive net value.

---

## 🚀 How to Run the Project

### Prerequisites
Make sure Python 3.9+ is installed.

```bash
# Clone the repository
git clone <your-repo-url>
cd "Super Selector RR"

# (Optional) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 1. Execute the Pipeline
Run `script.py` to process the ball-by-ball dataset, calculate the pressure indices, print terminal rankings, and export CSV files:

```bash
python script.py
```

Generated outputs:
- `output_batters.csv`: Complete ranked pool of qualified batters.
- `output_bowlers.csv`: Complete ranked pool of qualified bowlers.
- `output_validation.csv`: Situation split and hazard rate comparison.

### 2. Launch the Streamlit Dashboard
Launch the interactive web dashboard to explore charts, rankings, and player deep-dives:

```bash
streamlit run dashboard.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Repository Structure & GitHub Guidelines

### What to Add / Commit to GitHub ✅
Commit code, documentation, and small summary/output tables:
- `script.py`: Core processing pipeline and ranking algorithm.
- `dashboard.py`: Interactive Streamlit dashboard application.
- `output_batters.csv`: Top batter rankings and output metrics.
- `output_bowlers.csv`: Top bowler rankings and output metrics.
- `output_validation.csv`: Pressure regime validation table.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: Files and directories excluded from tracking.
- `README.md`: Methodology, rankings, and reproduction guide.

### What NOT to Add / Exclude from GitHub ❌
The following files are excluded via `.gitignore`:
- **Large raw datasets (`ball_by_ball.csv`)**: At ~39 MB, raw delivery logs bloat git repository history. If teammates need raw data, host it via Git LFS, cloud storage (Google Drive / S3 / Kaggle), or extract locally.
- **Virtual environments (`.venv/`, `venv/`, `env/`)**: Machine-specific environments.
- **Python cache (`__pycache__/`, `*.pyc`)**: Compiled Python bytecode.
- **Operating System artifacts (`.DS_Store`, `Thumbs.db`)**: OS-generated metadata.
- **Sensitive files (`.env`, private API keys)**: Never commit secrets.

---

## 🛠 Tech Stack
- **Data Engineering:** Python, Pandas, NumPy
- **Interactive Visualization:** Streamlit, Plotly Express, Plotly Graph Objects
- **Statistical Modeling:** Logistic Sigmoid, Empirical Bayes Shrinkage, Expectation Baselines
