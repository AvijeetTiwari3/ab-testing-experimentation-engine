# Experimentation Methodology Card: Dual-Engine A/B Testing Platform
### *Google Research & Large-Scale Experimentation Standard*

---

## 1. Experimentation Framework Overview
- **Name:** Statistical A/B Testing & Experimentation Analytics Engine
- **Architectural Paradigm:** Dual-Engine (Frequentist Hypothesis Testing + Bayesian Beta-Binomial Posterior Decision Theory)
- **Primary Objective:** Provide automated, mathematically rigorous, bias-free decision boundaries for product feature launches.
- **Reference Literature:** Kohavi, Tang, & Xu (2020), *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*, Cambridge University Press.

---

## 2. Statistical Machinery & Formulation

### A. Pre-Experiment Power Sizing
$$\text{Sample Size per Variant } n = \left[ \frac{z_{1-\alpha/2} \sqrt{2 \bar{p}(1 - \bar{p})} + z_{1-\beta} \sqrt{p_1(1-p_1) + p_2(1-p_2)}}{p_2 - p_1} \right]^2$$
- Guarantees $80\%$ statistical power to detect an absolute Minimum Detectable Effect ($\text{MDE}$) at significance level $\alpha = 0.05$.

### B. Frequentist Inference Engine
- **Test Statistic:** Two-Proportion Pooled Z-Test:
  $$Z = \frac{\hat{p}_{\text{trt}} - \hat{p}_{\text{ctrl}}}{\sqrt{\hat{p}_{\text{pool}}(1 - \hat{p}_{\text{pool}}) \left( \frac{1}{n_{\text{ctrl}}} + \frac{1}{n_{\text{trt}}} \right)}}$$
- **Effect Size:** Cohen's $h$ metric for standardized proportion distance:
  $$h = 2 \arcsin \sqrt{\hat{p}_{\text{trt}}} - 2 \arcsin \sqrt{\hat{p}_{\text{ctrl}}}$$

### C. Bayesian Decision Theory Engine
- **Prior Distribution:** $p \sim \text{Beta}(\alpha_0=1, \beta_0=1)$ (Uniform uninformative prior).
- **Likelihood:** $k \sim \text{Binomial}(n, p)$.
- **Conjugate Posterior:** $p | k, n \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$.
- **Expected Business Loss (Risk):**
  $$\text{Risk}(\text{Ship Treatment}) = \mathbb{E}_{\text{posterior}} \left[ \max(p_{\text{ctrl}} - p_{\text{trt}}, 0) \right]$$

### D. Sequential Alpha-Spending (Peeking Problem Solution)
- **Boundary Function:** O'Brien-Fleming type boundary:
  $$z_{\text{crit}}(t) = \frac{z_{\alpha/2}}{\sqrt{t}}, \quad \alpha(t) = 2 \left( 1 - \Phi(z_{\text{crit}}(t)) \right)$$
  where $t = \frac{n_t}{N_{\text{planned}}}$ represents the information fraction.

---

## 3. Data Integrity & Validation Gates
1. **Contamination Filter:** Automatic purge of cross-variant exposure leaks ($\text{group} \ne \text{page}$).
2. **First-Exposure Deduplication:** Protects against user novelty effect and reactivation bias.
3. **Sample Ratio Mismatch (SRM) Test:** Chi-square goodness-of-fit check on traffic allocations ($\chi^2 \text{ test}, \alpha = 0.01$).

---

## 4. Empirical Evaluation on Real Dataset (290,584 Verified Users)
- **Dataset Source:** Kaggle E-Commerce Landing Page Experiment (294,478 raw rows).
- **Control Baseline:** $12.04\%$ ($17,489 / 145,274$).
- **Treatment Observed:** $11.88\%$ ($17,264 / 145,310$).
- **Frequentist Result:** $Z = -1.3109$, $p = 0.1899$ (Null result confirmed; no statistically significant difference).
- **Bayesian Result:** $P(\text{Treatment} > \text{Control}) = 9.37\%$. Deploying Treatment introduces a $31\times$ higher risk of conversion loss.
- **Decision:** **RETAIN CONTROL**. Do not ship variant.
