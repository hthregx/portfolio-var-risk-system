# Baseline Case-Study Notes

## Purpose

This note documents the five target dates selected for the baseline
error-analysis case study.

The cases are derived from the canonical baseline case-study artifact.
The analysis is descriptive and does not infer external news or event
causes.

## Selection rule

Cases were selected using a deterministic rule rather than by visual
inspection of the figures.

The selection prioritizes:

- high-severity exceptions;
- overlapping exception episodes;
- the longest Historical exception cluster;
- representation of shared exceptions;
- representation of Historical-only exceptions;
- representation of EWMA-only exceptions.

The final set contains five target dates:

- C01: EWMA-only;
- C02: shared exception;
- C03: Historical-only;
- C04: shared exception;
- C05: Historical-only.

---

## C01 — 2023-09-25

**Forecast date:** 2023-09-22
**Exception type:** EWMA-only
**Selection reason:** top EWMA-only severity

### Realized return

Target return: **-3.101004%**

### Historical Simulation

- Quantile return: **-3.509365%**
- VaR: **3.509365%**
- Violation: **No**
- Exception severity: **0.000000%**
- Preceding VaR change: **0.000000 percentage points**
- Next VaR change: **0.000000 percentage points**

The realized return remained above the Historical return quantile, so
the observation was not a Historical violation.

### EWMA

- Quantile return: **-2.529980%**
- VaR: **2.529980%**
- Violation: **Yes**
- Exception severity: **0.571023%**
- Preceding VaR change: **+0.160881 percentage points**
- Next VaR change: **+0.222796 percentage points**

EWMA produced a less negative return threshold than Historical on this
target date. The realized loss crossed the EWMA threshold but remained
inside the Historical threshold.

The following EWMA forecast increased after the target return became
observable.

---

## C02 — 2025-04-03

**Forecast date:** 2025-04-02
**Exception type:** Shared
**Selection reason:** longest Historical exception cluster

### Realized return

Target return: **-6.983986%**

### Historical Simulation

- Quantile return: **-1.779316%**
- VaR: **1.779316%**
- Violation: **Yes**
- Exception severity: **5.204671%**
- Preceding VaR change: **0.000000 percentage points**
- Next VaR change: **+0.063523 percentage points**

### EWMA

- Quantile return: **-1.766910%**
- VaR: **1.766910%**
- Violation: **Yes**
- Exception severity: **5.217076%**
- Preceding VaR change: **-0.021227 percentage points**
- Next VaR change: **+1.527418 percentage points**

Both baseline models were violated by the realized return.

This is a high-severity shared exception. The next EWMA forecast shows
a substantially larger increase in VaR than the next Historical
forecast after the target return became observable.

---

## C03 — 2025-04-04

**Forecast date:** 2025-04-03
**Exception type:** Historical-only
**Selection reason:** longest Historical exception cluster

### Realized return

Target return: **-2.298635%**

### Historical Simulation

- Quantile return: **-1.842838%**
- VaR: **1.842838%**
- Violation: **Yes**
- Exception severity: **0.455797%**
- Preceding VaR change: **+0.063523 percentage points**
- Next VaR change: **0.000000 percentage points**

### EWMA

- Quantile return: **-3.294328%**
- VaR: **3.294328%**
- Violation: **No**
- Exception severity: **0.000000%**
- Preceding VaR change: **+1.527418 percentage points**
- Next VaR change: **+0.031204 percentage points**

The realized loss crossed the Historical threshold but remained above
the more conservative EWMA return quantile.

The large preceding increase in EWMA VaR followed the previous shared
exception. On this target date, that higher EWMA risk estimate was
sufficient to avoid an EWMA violation.

---

## C04 — 2025-04-08

**Forecast date:** 2025-04-04
**Exception type:** Shared
**Selection reason:** longest Historical exception cluster

### Realized return

Target return: **-6.929932%**

### Historical Simulation

- Quantile return: **-1.842838%**
- VaR: **1.842838%**
- Violation: **Yes**
- Exception severity: **5.087094%**
- Preceding VaR change: **0.000000 percentage points**
- Next VaR change: **+0.196736 percentage points**

### EWMA

- Quantile return: **-3.325532%**
- VaR: **3.325532%**
- Violation: **Yes**
- Exception severity: **3.604400%**
- Preceding VaR change: **+0.031204 percentage points**
- Next VaR change: **+0.939614 percentage points**

Both models were violated again on this target date.

EWMA entered the date with a larger VaR estimate than Historical and
therefore recorded a smaller exceedance. However, the realized loss was
large enough to exceed both forecast thresholds.

Both models increased their subsequent VaR forecasts, with the next
EWMA increase larger than the next Historical increase.

---

## C05 — 2025-04-09

**Forecast date:** 2025-04-08
**Exception type:** Historical-only
**Selection reason:** longest Historical exception cluster

### Realized return

Target return: **-4.229804%**

### Historical Simulation

- Quantile return: **-2.039574%**
- VaR: **2.039574%**
- Violation: **Yes**
- Exception severity: **2.190230%**
- Preceding VaR change: **+0.196736 percentage points**
- Next VaR change: **+0.134656 percentage points**

### EWMA

- Quantile return: **-4.265146%**
- VaR: **4.265146%**
- Violation: **No**
- Exception severity: **0.000000%**
- Preceding VaR change: **+0.939614 percentage points**
- Next VaR change: **+0.207472 percentage points**

Historical remained in violation on this date.

EWMA had increased its VaR substantially following the preceding
shared exception. Its return quantile was therefore more negative than
the realized return, and no EWMA violation occurred.

This case illustrates a difference in short-run risk response between
the two baseline methods within the selected exception episode.

---

## Case-study summary

| Case | Target date | Type | Historical violation | EWMA violation |
|---|---|---|---|---|
| C01 | 2023-09-25 | EWMA-only | No | Yes |
| C02 | 2025-04-03 | Shared | Yes | Yes |
| C03 | 2025-04-04 | Historical-only | Yes | No |
| C04 | 2025-04-08 | Shared | Yes | Yes |
| C05 | 2025-04-09 | Historical-only | Yes | No |

The selected cases provide representation of all three relevant
exception categories: shared, Historical-only, and EWMA-only.

The April 2025 cases also capture a multi-observation Historical
exception episode and allow the subsequent risk response of the two
baseline methods to be compared.

## Interpretation boundary

All interpretations above are descriptive and based only on the
forecast, realized-return, violation, severity, and adjacent VaR
movement fields contained in the case-study artifact.

No external news, market event, or causal explanation is inferred.

A "next VaR change" refers to the change in the subsequent forecast
after the current target return became observable. It must not be
interpreted as information that was available at the original forecast
origin.