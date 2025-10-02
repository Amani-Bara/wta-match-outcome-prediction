# Analysis of Women's Tennis Association (WTA) Match Outcomes

This project analyzes **WTA matches (2023–2024)** to explore how player characteristics, match conditions, and statistics affect match outcomes.  
We combined datasets, performed preprocessing, visualization, and applied machine learning models to predict outcomes and player performance.

---

## 👥 Team Members
- **Maram Alhusami** — Data Cleaning & Visualization, Report (Intro, Methodology)  
- **Bushra Alshehri** — ML Models & Evaluation, Report (Discussion, Conclusion)  
- **Amani Albarazi** — Data Analysis & Results, ML Models, Report (Results)  
- **Sara Imran** — ML Models, Report (Background & References)  

_All members contributed to the problem statement._

---

## 📊 Project Description
The project merges WTA match datasets from 2023 and 2024 to:  
- Analyze player demographics, performance stats, and match conditions.  
- Train ML models (Random Forest, Logistic Regression, SVM, etc.) to predict match outcomes.  
- Provide insights to improve training and match strategies in professional tennis.

---

## 📂 Report Table of Contents
1. Introduction  
2. Problem Statement & Background  
3. Data Description  
4. Data Visualization  
5. Dataset Merging  
6. Analysis  
7. Results  
8. Discussion  
9. References  

---

## ⚙️ Installation
### Requirements
- Python 3.8+
- Jupyter Notebook
- Anaconda (optional, recommended)

### Libraries
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

Install with:
```bash
pip install -r requirements.txt 
```
--- 
### 📂 Data

This project uses WTA match datasets (2023 & 2024).
Place them in a local data/ folder:

wta_matches_qual_itf_2023.csv

wta_matches_qual_itf_2024.csv



---

### 📖 CodeBook
The following table outlines the key variables used in the analysis:

| Variable Name | Data Type | Description |  
|------------------------|-----------|------------------------------------------------------------------|  
| match\_id | Integer | Unique identifier for each match |  
| tournament\_level | String | Level of the tournament (e.g., ITF, WTA) |  
| tourney\_date | Integer | Date of the tournament in Unix timestamp format |  
| winner\_id | Integer | Unique identifier for the match winner |  
| loser\_id | Integer | Unique identifier for the match loser |  
| winner\_ht | Float | Height of the winner in cm |  
| loser\_ht | Float | Height of the loser in cm |  
| winner\_age | Float | Age of the winner in years |  
| loser\_age | Float | Age of the loser in years |  
| surface\_type | String | Type of court surface (e.g., Clay, Grass, Hard) |  
| match\_duration | Float | Duration of the match in minutes |  
| first\_serve\_winner | Integer | Number of first serves made by the winner |  
| first\_serve\_loser | Integer | Number of first serves made by the loser |


---

### ▶️ Usage

To run the analysis, open Jupyter Notebook and execute the following steps:

1. Load the necessary libraries:

```bash 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

2. Load the dataset:
```bash
wta_matches = pd.read_csv('path/to/wta_matches_combined.csv')

```
3. Split data for model training and evaluation:
```bash
from sklearn.model_selection import train_test_split
X = wta_matches.drop('match_result', axis=1)  # Features
y = wta_matches['match_result']               # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

4. Implement machine learning models:
```bash
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

5. Evaluate model performance:
```bash
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy:.2f}')
```

---
### ✅ Conclusion

This README outlines the structure and execution of the project, detailing the contributions of each team member and the methodologies employed in the analysis of WTA match outcomes. The insights gained aim to enhance understanding and performance in competitive tennis.
