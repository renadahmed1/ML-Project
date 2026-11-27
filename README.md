# End-to-End Machine Learning Project

Classifying Raisin and Dry Bean varieties from shape features, comparing
Logistic Regression, SVM and Random Forest.

## Team
- Mohamed Abdalrasoul
- Renad Ahmed
- Maryam Tarek

## Task
Multi-class classification: predict the variety of a grain (target column `Class`)
from numeric shape features measured on its image.

## Datasets (both from the UCI Machine Learning Repository, license CC BY 4.0)
1. Raisin: 900 rows, 7 features, 2 classes (Kecimen, Besni)
   https://archive.ics.uci.edu/dataset/850/raisin
2. Dry Bean: 13,611 rows, 16 features, 7 classes
   https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

Both original Excel files are already included in the `data` folder.

## Methods
- Logistic Regression (linear model)
- SVM (kernel / margin based model)
- Random Forest (tree ensemble)

Preparation: check missing values, remove duplicate rows, label-encode the target,
median imputation, standard scaling, stratified 80/20 train/test split (random_state=42).
Tuning: GridSearchCV, 5-fold cross validation on the training set, scoring = macro F1.
Evaluation (same split and metrics for every method): accuracy, precision, recall,
F1 (macro) and confusion matrix.

## Project files
```
project.py            main code: runs the whole experiment end to end
data/                 the two datasets
results/              output tables (CSV) and figures (PNG)
report/Report.pdf     the scientific report
```

## How to run
```
pip install -r requirements.txt
python project.py
```
This prints the results for each dataset and saves the comparison tables, comparison
charts, class distribution charts and confusion matrices in `results/`.
It takes about 1-2 minutes.

To rebuild the report after running the code (needs `pip install reportlab`):
```
python report/make_report.py
```

## Main results (test set)
- Raisin: Logistic Regression 88.89%, SVM 87.78%, Random Forest 86.67% accuracy
- Dry Bean: SVM 92.36%, Logistic Regression 91.92%, Random Forest 91.92% accuracy
