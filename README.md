# Personal Project with Azure

Machine learning projects built while learning Azure ML.

## Project Structure

| Folder | Description |
|--------|-------------|
| `01-regression/` | Bike rental demand prediction using regression techniques |
| `02-classification/` | Wine variety & customer churn classification |
| `03-pipelines/` | Azure ML pipeline steps for churn prediction |
| `04-challenges/` | ML challenges (real estate price prediction) |
| `azure-ml-utils/` | Azure ML utility scripts (invoker, urldecode_invoker) |

## 01 - Regression
- **bike_rentals_basic.py** - Linear regression on bike share data
- **bike_rentals_multimodel.py** - Comparing Lasso, Decision Tree, Random Forest, Gradient Boosting, MLP
- **bike_rentals_optimize.py** - Hyperparameter tuning with GridSearchCV & pipelines

## 02 - Classification
- **wine_classification.py** - Multi-class wine variety prediction (6 classifiers + tuning)
- **churn_prediction.py** - Customer churn prediction with MLflow tracking

## 03 - Pipelines
- **churn_prep_step.py** - Data preprocessing pipeline step (scaling, encoding, split)
- **churn_train_step.py** - Training pipeline step (reads preprocessed data, trains & logs with MLflow)

## 04 - Challenges
- **real_estate_prediction.py** - Real estate price prediction with feature engineering & Gradient Boosting

## Azure ML Utils
- **invoker.py** - Azure ML module invoker (runs python modules via subprocess/runpy)
- **urldecode_invoker.py** - Same as invoker but with URL-decoding for encoded arguments

## Tech Stack
- Python, scikit-learn, pandas, numpy, matplotlib, seaborn
- Azure ML, MLflow