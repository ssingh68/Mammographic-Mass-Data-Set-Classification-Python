# Mammographic-Mass-Data-Set-Classification-Python

Data - https://www.kaggle.com/overratedgman/mammographic-mass-data-set/data

Mammography is the most effective method for breast cancer screening available today. However, the low positive predictive value of breast
biopsy resulting from mammogram interpretation leads to approximately 70% unnecessary biopsies with benign outcomes. To reduce the high
number of unnecessary breast biopsies, several computer-aided diagnosis (CAD) systems have been proposed in the last years.These systems
help physicians in their decision to perform a breast biopsy on a suspicious lesion seen in a mammogram or to perform a short term follow-up examination instead.

This data set is used to predict whether a mammogram mass is benign or malignant.

This data contains 961 instances of masses detected in mammograms, and contains the following attributes:

1. BI-RADS assessment: 1 to 5 (ordinal)

2. Age: patient's age in years (integer)

3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)

4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)

5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)

6. Severity: benign=No or malignant=Yes (binary)

### Models used - Decision Tree using Test/Train split, Decision Tree using 10-fold CV, Random Forest, KNeighborsClassifier, KNeighborsClassifier using K values ranging from 1 to 50 & Multinomial Naive Bayes model.
