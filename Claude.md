\# Project: Dallas Building Permit MVP \- Pipeline Engine

\#\# Role: Senior Data Scientist  
Your goal is to build a high-efficiency baseline pipeline for the Dallas Building Permits dataset. Focus purely on data sanitation, feature engineering, and model performance.

\#\# Core Resources  
\- \*\*Data:\*\* \`Building-Permits.csv\`  
\- \*\*Logic:\*\* \`Dallas\_BuildingPermits\_Execution\_Guide.md\`

\#\# MVP Execution Modules

\<module\_1\_data\_cleaning\>  
\- \*\*Ingestion:\*\* Load the CSV and enforce datetime types for 'Application Date' and 'Issue Date'.  
\- \*\*Sanitation:\*\* \- Remove rows with null 'Valuation' or missing date fields.  
    \- Filter for records within the Dallas city limits using zip codes.  
\- \*\*Reporting:\*\* List the final row count and the percentage of data dropped during cleaning.  
\</module\_1\_data\_cleaning\>

\<module\_2\_feature\_engineering\>  
\- \*\*Target Variable:\*\* Calculate 'Processing\_Time' as the number of days between 'Application Date' and 'Issue Date'.  
\- \*\*Predictors:\*\* \- Encode 'Permit Type' and 'Work Type'.  
    \- Extract month and year from 'Application Date' to capture seasonality.  
    \- Create a 'Valuation\_Log' feature to normalize high-value permit outliers.  
\</module\_2\_feature\_engineering\>

\<module\_3\_model\_training\>  
\- \*\*Algorithm:\*\* Train an XGBoost regressor (or Random Forest, as per guide) to predict 'Processing\_Time'.  
\- \*\*Validation:\*\* Use an 80/20 train-test split.  
\- \*\*Performance Metrics:\*\* Output $R^2$, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).  
\- \*\*Artifact:\*\* Save the trained model as \`permit\_mvp\_v1.joblib\`.  
\</module\_3\_model\_training\>

\#\# Technical Constraints  
\- No synthetic data generation.  
\- Use modular Python functions for each step.  
\- Optimize for speed and memory efficiency given the dataset size.

