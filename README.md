  
\# Project Title: Analysis of Women's Tennis Association (WTA) Match Outcomes

\#\# Team Members  
1\. \*\*Maram Alhusami\*\*  
\- Proposal: Data Sets Merging, Preparation and Data Visualization
\- Presentation:  Slides 1-10 
\- Report: Introduction & Methodology  
\- Code: Data Cleaning and Prepration, Data Visualization and Merging Sets
   
3\. \*\*Bushra Alshehri\*\*  
\- Proposal: Outcome Variables  
\- Presentation: Slides 11-21
\- Report: Discussion and Conclusions  
\- Code: Machine Learning and Model Implementation and Evaluation

2\. \*\*Amani Albarazi\*\*  
\- Proposal: Predictor Variables  
\- Presentation:Slides 22-32
\- Report: Data Analysis & Results  
\- Code: Machine Learning and Model Implementation and Evaluation

4\. \*\*Sara Imran\*\*  
\- Proposal: Predictor Variables  
\- Presentation: Slides 32-42
\- Report: Background & References  
\- Code: Machine Learning and Model Implementation and Evaluation

All team members contributed to the \*\*High-Level Statement of the Problem\*\* in the Proposal section.

\#\# Project Description  
The "Analysis of WTA Match Outcomes" project combines datasets from the WTA matches in 2023 and 2024 to analyze player performance and 
match results. By evaluating various statistical measures, player demographics, and match conditions, the project aims to uncover patterns 
that influence match outcomes. The findings will provide insights for improving player training and match strategies in professional tennis.

\#\# Table of Contents for Report  
1\. Introduction  
2\. Problem Statement and Background  
3\. Data Description 
4\. Data Visualization 
5\. Data Sets Merging
6\. Analysis  
7\. Results  
8\. Discussion  
9\. References

\#\# Installation  
To run this project, ensure you have the following software installed:

\- \*\*Python\*\*: 3.8  
\- \*\*Anaconda\*\*: Latest version (e.g., 2023.x)  
\- \*\*Jupyter Notebook\*\*: Latest version

\#\#\# Libraries  
\- \*\*Pandas\*\*: 1.3.5  
\- \*\*Seaborn\*\*: 0.11.2  
\- \*\*Matplotlib\*\*: 3.4.3  
\- \*\*NumPy\*\*: 1.21.5  
\- \*\*Scikit-learn\*\*: 1.0.2

\#\# Codebook  
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

\#\# Usage  
To run the analysis, open Jupyter Notebook and execute the following steps:

1\. Load the necessary libraries:  
\`\`\`python  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
\`\`\`

2\. Load the dataset:  
\`\`\`python  
wta\_matches \= pd.read\_csv('path/to/wta\_matches\_combined.csv')  
\`\`\`

3\. Split data for model training and evaluation:  
\`\`\`python  
from sklearn.model\_selection import train\_test\_split  
X \= wta\_matches.drop('match\_result', axis=1) \# Features  
y \= wta\_matches\['match\_result'\] \# Target variable  
X\_train, X\_test, y\_train, y\_test \= train\_test\_split(X, y, test\_size=0.2, random\_state=42)  
\`\`\`

4\. Implement machine learning models:  
\`\`\`python  
from sklearn.ensemble import RandomForestClassifier  
model \= RandomForestClassifier()  
model.fit(X\_train, y\_train)  
predictions \= model.predict(X\_test)  
\`\`\`

5\. Evaluate model performance:  
\`\`\`python  
from sklearn.metrics import accuracy\_score  
accuracy \= accuracy\_score(y\_test, predictions)  
print(f'Model Accuracy: {accuracy:.2f}')  
\`\`\`

\#\# Conclusion  
This README outlines the structure and execution of the project, detailing the contributions of each team member and the methodologies
employed in the analysis of WTA match outcomes. The insights gained aim to enhance understanding and performance in competitive tennis.  
