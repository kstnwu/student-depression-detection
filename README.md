# Student Depression Risk: Logistic Regression and Decision Analysis

Predict which university students are at risk of depression, then evaluate how those predictions should be used when a counseling center has limited capacity for outreach.

The focus is not on maximizing accuracy but on the decision layer: interpreting the model, choosing a threshold based on the costs of errors, and prioritizing students when only a fixed share can be contacted.

## Key Results

| | |
|---|---|
| Test AUC | 0.693 |
| Accuracy at 0.5 threshold | 0.729 (baseline: 0.731) |
| Cost reduction, cost-based threshold vs. 0.5 | 34% |
| Lift when contacting the top 10% of students | 1.83× vs. random |

- **Accuracy is misleading.** At the default 0.5 threshold, the model does not beat the majority-class baseline and identifies only 19% of depressed students.
- **A cost-based threshold changes the picture.** Assuming a missed depressed student costs 4 times an unnecessary contact, the optimal threshold is 0.2, raising the detection rate to 83% and cutting total cost by 34%.
- **Ranking adds the most value under tight capacity.** Contacting the highest-risk 5% of students reaches about 2 times as many depressed students as random selection.

## Data

[Student Depression Detection Dataset](https://www.kaggle.com/datasets/mkhoirulmustaqim/student-depression-detection-dataset) on Kaggle, originally published on Mendeley Data by Obaidul Haque and Samiul Alom. It contains 4,464 survey responses from university students in Bangladesh with 12 demographic, academic, social, and economic features and a binary depression label (26.9% positive).

**Data sanity check** (`00_data_check.ipynb`): `Health_condition` alone achieves an in-sample AUC of 0.949. Every student with *Critical* health is labeled depressed and none with *Good* health are, indicating the label is largely rule-based on this feature. It is excluded from modeling to avoid leakage. No other feature shows a deterministic pattern.

## Approach

1. **Logistic regression** with `statsmodels` GLM (binomial family, logit link) on a stratified 70/30 train/test split
2. **Coefficient interpretation** through odds ratios and 95% confidence intervals
3. **Manual prediction** for a single student: logit → odds → probability, verified against `predict()`
4. **Evaluation** with a confusion matrix, majority-class baseline, ROC/AUC, and a calibration curve
5. **Cost-based threshold** derived from assumed error costs and validated on the test set
6. **Top-k selection** compared with random selection under fixed outreach capacity

## Findings

### Risk factors

Holding other variables constant:

| Feature | Odds ratio | Interpretation |
|---|---|---|
| Single (vs. in a relationship) | 5.08 | 5 times the odds of depression |
| Unmarried (vs. married) | 2.05 | about 2 times the odds |
| Living cost, per 1,000 units | 1.07 | 7% higher odds |
| Male (vs. female) | 0.58 | 42% lower odds |
| CGPA, per 1 point | 0.49 | 51% lower odds |
| Active friend circle | 0.67 | 33% lower odds |
| Rich family (vs. lower-middle class) | 0.23 | 77% lower odds |

Age, semester, living arrangement, and home division are not significant at the 5% level.

### Ranking and calibration

<p>
  <img src="figures/roc_curve.png" width="45%" />
  <img src="figures/calibration_curve.png" width="45%" />
</p>

The model ranks students moderately well (AUC = 0.693). The 0.5 threshold sits in the lower-left of the ROC curve, leaving substantial room to detect more depressed students at the cost of more false positives. Predicted probabilities are reasonably well calibrated, which supports reading them as risk estimates when setting a cost-based threshold.

AUC is identical when computed on probabilities, logits, or odds, since these are monotonic transformations that preserve the ranking of students.

### Choosing a threshold

<img src="figures/cost_vs_threshold.png" width="70%" />

With C_FP = 1 and C_FN = 4, contacting a student is worthwhile when the predicted probability exceeds C_FP / (C_FP + C_FN) = 0.2. The test-set cost curve is flat between about 0.11 and 0.25, and the theoretical threshold falls within this range.

| Threshold | Accuracy | TPR | FPR | Students contacted | Total cost |
|---|---|---|---|---|---|
| 0.5 | 0.729 | 0.19 | 0.07 | 142 | 1,236 |
| 0.2 | 0.526 | 0.83 | 0.59 | 874 | 818 |

Accuracy falls while total cost improves, because accuracy treats both types of error as equally costly.

### Prioritizing under limited capacity

<img src="figures/cumulative_gains.png" width="50%" />

A 0.2 threshold flags 65% of students, which may exceed counseling capacity. When only k% of students can be contacted, ranking by predicted risk outperforms random selection:

| Capacity | Depressed reached (model) | Expected (random) | Lift |
|---|---|---|---|
| 5% | 36 | 18.0 | 1.99 |
| 10% | 66 | 36.1 | 1.83 |
| 20% | 128 | 72.2 | 1.77 |
| 30% | 171 | 108.3 | 1.58 |
| 50% | 250 | 180.5 | 1.39 |

At 20% capacity, the same counseling resources reach 35% of depressed students instead of 20%.

## Limitations

- The data comes from university students in Bangladesh and may not generalize to other populations.
- How the depression label was measured is not documented, and its strong dependence on `Health_condition` suggests it may be partly constructed.
- The 4:1 cost ratio is an assumption and should be set together with the counseling center.
- The model is intended to prioritize outreach, not to diagnose depression.

## Repository Structure

```
├── data/                          # raw data (not tracked; download from Kaggle)
├── figures/                       # plots used in this README
├── notebooks/
│   ├── 00_data_check.ipynb        # data sanity check and leakage screening
│   └── 01_classification.ipynb    # modeling, evaluation, and decision analysis
└── README.md
```

## How to Run

1. Download `Student_Depression_Dataset.csv` from Kaggle and place it in `data/`.
2. Install dependencies: `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `matplotlib`.
3. Run the notebooks in order from the `notebooks/` folder.
