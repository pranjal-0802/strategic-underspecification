# Robustness Report: Regularity Assumption (Assumption 1) Characterization

**Verdict:** **CAUTION** (Holds in 10.1% of plausible parameter space; restrictive when specification costs are low or accuracy is high)

### Mathematical Definition (Assumption 1):
$$\text{In the relevant range of } m, \quad (k - m) \ln q(a, g) \le -1$$

Since $\ln q \approx -(1 - q)$ for $q \approx 1$, this condition is approximately:
$$(k - m)(1 - q) \ge 1$$
which requires that the expected number of wrong attributes under silent guessing is at least 1.

### Grid Audit Results ($k \in [5, 15], q \in [0.70, 0.95], \kappa \in [0.20, 2.00]$):
- **Overall Satisfaction Rate:** **10.1%** (73 / 726 grid points).
- **Breakdown by User Specification Cost $\kappa$:**
  - $\kappa = 0.20$: **0.0%** satisfaction
  - $\kappa = 0.40$: **0.0%** satisfaction
  - $\kappa = 0.60$: **0.0%** satisfaction
  - $\kappa = 0.80$: **2.5%** satisfaction
  - $\kappa = 1.00$: **13.2%** satisfaction
  - $\kappa = 2.00$: **44.6%** satisfaction

### Analytical Diagnosis:
1. **Why it fails when $\kappa \le 0.60$:**
   When user specification cost $\kappa$ is small, the user finds it optimal to specify nearly all attributes ($m^* \to k$).
   Consequently, the number of unspecified attributes $k - m^*$ approaches 0.
   Because $k - m^* \approx 0$, $(k - m^*)\ln q \approx 0 > -1$, violating the assumption.
2. **Why it fails when $q \ge 0.90$:**
   When accuracy $q$ is high, $|\ln q|$ is small (e.g., $|\ln 0.95| \approx 0.051$).
   Satisfying $(k - m^*) \ln q \le -1$ requires $k - m^* \ge 1 / 0.051 \approx 19.6$ unspecified attributes, which exceeds the entire attribute budget $k \le 15$ in the paper's working range.
3. **Where it holds:**
   The assumption holds strictly when $\kappa \ge 1.0$ (users have high specification cost and leave attributes unspecified) and $q \le 0.85$ (guessing is noticeably noisy).

### Recommendation for Paper:
The paper's updated Remark 1 is fully confirmed by this audit: Assumption 1 is a sufficient condition for Topkis's theorem, but should be understood as restrictive for highly capable models ($q > 0.90$) or low-cost users ($\kappa < 0.60$).

**Artifacts Generated:**
- CSV: `assumption1_regularity.csv`
- Figure: `assumption1_regularity.png`
- Summary: `assumption1_regularity.md`
