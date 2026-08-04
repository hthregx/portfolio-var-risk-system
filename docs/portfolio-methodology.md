# Portfolio Return Methodology

## 1. Portfolio Definition

The portfolio module models a portfolio containing the following assets:

- HPG
- FPT
- MWG

The default portfolio uses equal weights across all included assets.

The module is responsible for:

- creating portfolio weights;
- validating portfolio weights;
- aggregating individual asset returns;
- converting portfolio simple returns into portfolio log returns.

The module does not perform portfolio optimization or final Value at Risk estimation.

---

## 2. Weight Convention

For a portfolio containing \(N\) assets, the equal weight assigned to each asset is:

\[
w_i = \frac{1}{N}
\]

For the three-asset portfolio HPG, FPT and MWG:

\[
w_{HPG} = w_{FPT} = w_{MWG} = \frac{1}{3}
\]

Portfolio weights must satisfy:

\[
\sum_{i=1}^{N} w_i = 1
\]

The current project assumes a long-only portfolio:

\[
w_i \geq 0
\]

Therefore, negative portfolio weights are not allowed.

Weights containing missing or infinite values are also invalid.

---

## 3. Simple Return Formula

The simple return of asset \(i\) at trading date \(t\) is calculated as:

\[
r_{i,t}
=
\frac{P_{i,t}}{P_{i,t-1}} - 1
\]

where:

- \(P_{i,t}\) is the closing price of asset \(i\) at date \(t\);
- \(P_{i,t-1}\) is the previous trading day's closing price.

Simple returns are used as the input for portfolio aggregation.

---

## 4. Portfolio Aggregation

The portfolio simple return at date \(t\) is calculated as:

\[
R_{p,t}
=
\sum_{i=1}^{N} w_i r_{i,t}
\]

where:

- \(w_i\) is the portfolio weight of asset \(i\);
- \(r_{i,t}\) is the simple return of asset \(i\) at date \(t\).

For an equal-weight portfolio containing HPG, FPT and MWG:

\[
R_{p,t}
=
\frac{
r_{HPG,t}
+
r_{FPT,t}
+
r_{MWG,t}
}{3}
\]

Example:

```text
HPG return = 0.03
FPT return = 0.00
MWG return = -0.03