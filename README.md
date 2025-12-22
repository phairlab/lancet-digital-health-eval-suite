# PHAIR Model Evaluation Suite: Medical Binary Classification 


## Introduction

This repo provides quick, rigorous evaluation of clinical prediction models following the core recommendations from Van Calster et al.'s 2025 *Lancet Digital Health* Viewpoint paper. Given predicted probabilities and true outcomes, it generates the four essential plots/numbers: AUROC, calibration curve, decision curve analysis, and risk distributions.

Performance metric selection for clinical ML models is contentious—researchers disagree on which measures are appropriate. This repo implements recommendations from a comprehensive expert consensus paper, giving you a defensible, citable rationale for your evaluation approach.

## Core Metrics (Van Calster et al. 2025)

1. **AUROC** - Discrimination performance
2. **Calibration Plot** - Agreement between predictions and observations
3. **Decision Curve** - Net benefit across decision thresholds
4. **Risk Distribution** - Probability distributions by outcome


## Quickstart

```python
from lda_eval import evaluate_model
import numpy as np

# Your model predictions
y_true = np.array([0, 1, 1, 0, ...])  # Binary outcomes
y_prob = np.array([0.2, 0.8, 0.6, ...])  # Predicted probabilities

# Generate all recommended plots
evaluate_model(y_true, y_prob, output_dir="results/")
```


## Example Usage

### Set Up Dependencies
```bash
pip install -r requirements.txt
```

### Import as Package for Direct Use
```python
from lda_eval import evaluate_model

# Individual plots
auroc(y_true, y_prob)
calibration(y_true, y_prob)
decision_curve(y_true, y_prob, threshold_range=(0.0, 0.5))
risk_distribution(y_true, y_prob)

# All at once
evaluate_model(y_true, y_prob)
```

### Command Line with Results Directory (Recommended for Cross-Validation)

In your original code that trains or generates inference from the model, insert the following code snippet to save the true classes and output probabilities in JSON form.

```python
import json

# TRAINING LOOP
for fold_num in range(n_folds):

    ##### ...
    ##### Training logic 
    ##### ...

    # EXAMPLE: FETCH MODEL PROBABILITIES
    probas_ = classifier.predict_proba(X_test)

    # VERIFY OUTPUT DIRECTORY
    results_folder_path = "experiment_results"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # SAVE TRUE CLASSES AND PROBABILISTIC PREDICTIONS FROM MODEL
    test_predictions = {
        'y_true': y_test.tolist(),
        'y_proba': probas_[:, 1].tolist()
    }
    with open(f"{results_folder_path}/f{fold_num}_predictions.json", 'w') as f:
        json.dump(test_predictions, f, indent=4)
```

After running your script, the analysis can be run by passing your output folder into `ldh_eval.py`. This saves the in-fold plots and numbers in each individual fold's sub-folder, and meta-analysis of all folds into the `input_dir` path.
 
```bash
python ldh_eval.py --input_dir "experiment_results/"
```


## Citation

Van Calster B, Collins GS, Vickers AJ, Wynants L, Kerr KF, Barreñada L, Varoquaux G, Singh K, Moons KGM, Hernandez-Boussard T, Timmerman D, McLernon DJ, van Smeden M, Steyerberg EW, on behalf of Topic Group 6 of the STRATOS initiative. Evaluation of performance measures in predictive artificial intelligence models to support medical decisions: overview and guidance. *Lancet Digit Health* 2025. https://doi.org/10.1016/j.landig.2025.100916