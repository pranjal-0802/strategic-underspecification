# Robustness Report: $\mu_A$ Comparative Statics & Bias Confounding

**Verdict:** **PASS** (Remark rmk:mu-lambda analytically and numerically verified)

### 1. Mathematical Derivation of the Confound:
From Proposition 4, when $\Delta\bar{\gamma} > 0$, the interior pooling ask rate is:
$$a^{SE} = -\frac{C_0 \bar{\gamma} + \Delta \bar{R}_0}{2\Delta\bar{\gamma}} = -\frac{\bar{R}_0}{2\bar{\gamma}} - \frac{\mu_A \Lambda}{2(\lambda_A - \mu_A \Lambda)} = -\frac{\bar{R}_0}{2\bar{\gamma}} - \frac{1}{2} \left[ \frac{1}{\frac{\lambda_A}{\mu_A \Lambda} - 1} \right]$$

Notice that $a^{SE}$ depends on the two bias parameters $(\mu_A, \lambda_A)$ **strictly through the scalar ratio $\lambda_A / \mu_A$**!

### 2. Exact Derivative Ratio:
Differentiating directly:
$$\frac{\partial a^{SE}}{\partial \lambda_A} = \frac{\mu_A \Lambda}{2(\lambda_A - \mu_A \Lambda)^2} = \frac{\mu_A \Lambda}{2\Delta^2}$$
$$\frac{\partial a^{SE}}{\partial \mu_A} = -\frac{\lambda_A \Lambda}{2(\lambda_A - \mu_A \Lambda)^2} = -\frac{\lambda_A \Lambda}{2\Delta^2}$$

Taking the ratio:
$$\frac{\partial a^{SE} / \partial \mu_A}{\partial a^{SE} / \partial \lambda_A} = -\frac{\lambda_A}{\mu_A}$$

- **Numerical Verification:** Across the parameter sweep, the finite-difference ratio matches $-\lambda_A / \mu_A$ with maximum error `2.66e-08`.

### 3. Empirical Implications for Field Identification:
1. **Opposite Derivative Signs, Identical Bias Effect:**
   - Under-asking bias from friction corresponds to $\lambda_A > c_Q \implies d\lambda_A > 0$.
   - Under-asking bias from discounting corresponds to $\mu_A < 1 \implies d\mu_A < 0$.
   - Because $\frac{\partial a^{SE}}{\partial \mu_A} < 0$, a decrease in $\mu_A$ ($d\mu_A < 0$) produces $da^{SE} = \frac{\partial a^{SE}}{\partial \mu_A} d\mu_A > 0$, exactly matching the directional effect of increasing $\lambda_A$!
2. **Total Observational Confound:**
   - As shown by the level curves in the contour plot, any pair $(\mu_A, \lambda_A)$ sharing the same ray $\lambda_A / \mu_A$ yields the exact same pooling ask rate $a^{SE}$.
   - **Conclusion:** Observing $a^{SE}$ in field data **cannot separately identify** whether an AI assistant's under-asking behavior originates from friction misperception ($\lambda_A > c_Q$) or downstream welfare discounting ($\mu_A < 1$). An independent instrument or cost measurement is required.

**Artifacts Generated:**
- CSV: `mu_comparative_statics.csv`
- Figure: `mu_comparative_statics.png`
- Summary: `mu_comparative_statics.md`
