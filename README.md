# Predicting Graduate Admissions

This project predicts the likelihood of graduate admissions using machine learning techniques. By analyzing data like GRE scores, TOEFL scores, GPA, and other features, we build and evaluate models to predict a candidate's admission chances.
________________________________________
**Overview**
The project involves:
1.	Data Analysis: Understanding patterns and relationships in the dataset.
2.	Model Training: Training machine learning models to predict admission chances.
3.	Model Evaluation: Comparing models for performance and accuracy.
4.	Outputs: Predictions based on test data.
________________________________________
**Dataset**
The project uses the original_data.csv dataset, which contains the following features:
- GRE Score
- TOEFL Score
- University Rating
- Statement of Purpose (SOP) score
- Letter of Recommendation (LOR) score
- Undergraduate GPA (CGPA)
- Research experience (binary: 0 or 1)
- Chance of Admission (target variable)
________________________________________
**Requirements**
The project is implemented using Python. Required libraries include:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
Install dependencies using:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```
________________________________________
**Setup and Execution**
1.	Clone the Repository:
2.	git clone https://github.com/SudhaChandrikaY/PredictingGraduateAdmissions.git
3.	cd PredictingGraduateAdmissions
4.	Run the Notebook: Open the Jupyter notebook MachineLearningProject.ipynb:
5.	jupyter notebook MachineLearningProject.ipynb
6.	Execution:
    - Load and preprocess the dataset.
    - Train various machine learning models (e.g., Linear Regression).
    - Evaluate models using metrics like R² score and Mean Squared Error.
________________________________________
**Project Files**
- MachineLearningProject.ipynb: Contains the full implementation:
  - Data loading and preprocessing.
  - Exploratory Data Analysis (EDA) using graphs.
  - Model training and evaluation.
- original_data.csv: The dataset used for training and testing.
- score.csv: File storing model outputs/predictions.
________________________________________
**Key Visualizations**
- Correlation Heatmaps: Displays relationships between features.
- Scatter Plots: GRE scores, GPA, and their impact on admission chances.
- Model Metrics: Compare predictions with actual data using evaluation graphs.
________________________________________
**Results**
The models predict chances of admission based on input features. Key findings:
- GRE, TOEFL, and GPA have strong correlations with admission chances.
- Linear Regression performs effectively with good R² scores.
________________________________________
**Future Work**
- Include advanced models like Random Forest, XGBoost, or Neural Networks.
- Experiment with feature engineering and scaling for better accuracy.
- Deploy the model as a web app for real-world usage.
________________________________________
**Author**

Developed by SudhaChandrikaY.



