# C-TAEA Replication Issues Report

## Scope

This document summarizes implementation and experiment issues identified during a paper-to-code review of the CTAEA folder and the main paper PDF.

## Executive Summary

Replication is currently not faithful due to several high-impact mismatches in algorithm behavior and metric computation. The most critical issues are:

1. Restricted mating selection does not follow Algorithm 4.
2. Monte Carlo hypervolume uses reversed dominance logic.
3. Many-objective weight vector construction does not match Table IV behavior.
4. Some decision-space constrained PF reference samplers are approximate/inconsistent.

These issues can materially alter both absolute and comparative results.

## Findings

### 1) Critical: Restricted mating selection mismatch with Algorithm 4

Severity: Critical

What is wrong:
- Offspring generation selects both parents with the same method for parent 1 logic.
- Parent 2 should be selected probabilistically from CA or DA using rho_c.
- A dedicated function for second parent exists but is never used.

Evidence:
- CTAEA/Algorithms/ctaea.py:170-172
- CTAEA/Algorithms/ctaea.py:190-217
- CTAEA/Algorithms/ctaea.py:219-232

Paper expectation:
- Algorithm 4 defines:
  - p1 from CA if rho_c > rho_d else DA
  - p2 from CA with probability rho_c else DA

Replication impact:
- Changes CA/DA collaboration dynamics and search behavior.

Recommended fix:
- Use separate logic for p1 and p2 in offspring generation.
- Call second-parent routine for p2 and ensure rho_c is recomputed from current Hm.

---

### 2) Critical: Monte Carlo HV dominance test is reversed

Severity: Critical

What is wrong:
- Current MC HV marks a sample as dominated when sample <= point.
- For minimization, dominated by a point should be point <= sample.

Evidence:
- CTAEA/Metrics/hv.py:180-181

Sanity check:
- Single 6D point p=(0.2,...,0.2), ref=(1,...,1)
- Expected HV is product(1-0.2)=0.262144
- Current implementation returned 0.0 in test

Replication impact:
- High-dimensional HV values are invalid when MC fallback is used.

Recommended fix:
- Replace dominance condition with point <= sample component-wise.

---

### 3) High: Weight vector / population mapping does not match many-objective setup

Severity: High

What is wrong:
- Table values specify N and H for m in {3,5,8,10,15}.
- Current generation uses single-layer Das-Dennis at H, then randomly subsamples to N when counts mismatch.
- For m=8,10,15 this produces very large sets before random truncation.

Evidence:
- CTAEA/Algorithms/utils.py:41-54
- CTAEA/Algorithms/utils.py:17-37
- CTAEA/Algorithms/ctaea.py:61-70

Observed generated counts (single layer):
- m=8, H=4 -> 330 (table N is 156)
- m=10, H=4 -> 715 (table N is 275)
- m=15, H=3 -> 680 (table N is 135)

Replication impact:
- Subregion geometry and distribution pressure differ from reported setup.

Recommended fix:
- Implement proper layered/reference-vector construction for many-objective cases instead of random subsampling.

---

### 4) High: PF reference generation for some DC problems is approximate/inconsistent

Severity: High

What is wrong:
- Some PF samplers reconstruct decision variables from objective values using assumptions marked as not generally valid.
- This can corrupt reference fronts used for IGD.

Evidence:
- CTAEA/Problems/DC1DTLZ1.py:76-87
- CTAEA/Problems/DC3DTLZ1.py:103-113

Replication impact:
- IGD values may not be directly comparable to paper results for affected DC cases.

Recommended fix:
- Generate PF references from problem-consistent parameterizations in decision space, then map to objective space.
- Validate feasibility filters against exact constraint definitions.

---

### 5) Medium: Peer methods are simplified baselines

Severity: Medium

What is wrong:
- Baseline algorithms are explicitly labeled simplified.
- This is acceptable for internal comparison, but not strict paper-level reproduction unless clearly disclosed.

Evidence:
- CTAEA/Algorithms/peer_algorithms.py:5-9

Replication impact:
- Relative ranking against literature baselines may differ from published tables.

Recommended fix:
- Either implement fuller baseline fidelity or label results as approximate baseline comparisons.

---

### 6) Medium: Non-deterministic seed offset due to Python hash randomization

Severity: Medium

What is wrong:
- Seed formula uses Python hash on strings.
- Hash values vary across processes unless PYTHONHASHSEED is fixed.

Evidence:
- CTAEA/Experiments/runner.py:210-211

Replication impact:
- Re-running experiments can change random seeds and outcomes unexpectedly.

Recommended fix:
- Replace hash with stable deterministic mapping (for example, fixed dictionary or stable string hashing via hashlib).

## Priority Fix Order

1. Fix restricted mating parent selection (Algorithm 4 fidelity).
2. Fix Monte Carlo HV dominance orientation.
3. Replace unstable seed construction with deterministic seed mapping.
4. Implement proper many-objective reference-vector construction.
5. Correct DC PF reference generation for IGD.
6. Improve or clearly disclaim simplified peer baselines.

## Validation Plan After Fixes

1. Unit checks
- Mating selection path test confirms p2 source probability tracks rho_c.
- HV MC single-point sanity tests in 3D and 6D.
- Deterministic seed test across multiple Python processes.

2. Small experiment checks
- Run m=3 on C1DTLZ3 and C2DTLZ2 for 3-5 runs.
- Run m=8 on one C and one DC problem to validate vector setup and HV behavior.

3. Full replication run
- 51 runs across full suite only after all above pass.

