# Health Insurance Smart Analysis

A Python machine learning project for smart analysis of health insurance services and branch recommendation.

This project generates synthetic health insurance data and applies different machine learning and deep learning techniques to analyze service cost, customer satisfaction, waiting time, branch score, suitable branch prediction, user risk level, clustering, and recommendation of better insurance branches.

## Project Goal

The main goal of this project is to demonstrate how supervised learning, unsupervised learning, dimensionality reduction, clustering, recommendation systems, and deep learning can be applied to a realistic health insurance scenario.

The project can be useful for students and beginner machine learning developers who want to learn how to build a complete ML pipeline from data generation to model evaluation and result export.

## Main Features

* Synthetic health insurance dataset generation
* Regression analysis for numerical targets
* Classification analysis for categorical targets
* PCA for dimensionality reduction
* Clustering with multiple algorithms
* Best K analysis for K-Means
* KNN-based branch recommendation
* Deep learning regression and classification using TensorFlow/Keras
* Autoencoder-based unsupervised learning
* Cross-validation
* CSV export for all important results

## Dataset

The dataset is generated inside the project using Python. It includes features such as:

* Age
* Annual visits
* Distance to branch
* Travel time
* Service cost
* Service type
* Insurance type
* Branch ID
* Branch quality
* Branch workload
* Waiting time
* Satisfaction score
* User comment
* Suitable branch
* Branch score
* Satisfaction level
* Risk level

The project generates datasets in different sizes:

```python
DATASET_SIZES = [300, 3000, 3000000]
```

For very large datasets, the project uses sampling to avoid excessive memory usage on normal laptops.

## Machine Learning Algorithms Used

### Supervised Regression

The project predicts numerical targets such as:

* Service cost
* Satisfaction score
* Waiting time
* Branch score

Algorithms used:

* Linear Regression
* Polynomial Regression + Ridge
* Ridge Regression
* Lasso Regression
* Decision Tree Regression
* Random Forest Regression
* KNN Regression
* Keras Neural Network Regression

Evaluation metrics:

* MAE
* RMSE
* R2 Score
* Overfitting gap
* Training time

### Supervised Classification

The project predicts classification targets such as:

* Suitable branch
* Satisfaction level
* Risk level

Algorithms used:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* KNN Classifier
* SVM Classifier
* Keras Neural Network Classifier

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Overfitting gap
* Training time

### Unsupervised Learning

The project also applies unsupervised learning methods such as:

* PCA
* K-Means
* MiniBatchKMeans
* BIRCH
* Gaussian Mixture Model
* Agglomerative Clustering
* DBSCAN
* Keras Autoencoder + KMeans

Clustering evaluation metrics:

* Silhouette Score
* Calinski-Harabasz Score
* Davies-Bouldin Score
* Number of clusters
* Noise ratio
* AIC and BIC for Gaussian Mixture Model

## Branch Recommendation System

The project includes a KNN-based branch recommendation system.

It uses similar users and their satisfaction results to recommend a more suitable branch for a new or existing insured user.

The recommendation system considers:

* Similar user behavior
* Branch satisfaction score
* Branch quality
* Suitable branch label
* Service type
* Insurance type
* Waiting time
* Travel time

## Output Files

After running the project, multiple CSV files are generated automatically:

```text
exercise5_health_insurance_dataset_300.csv
exercise5_health_insurance_dataset_3000.csv
exercise5_health_insurance_dataset_3000000.csv

exercise5_sklearn_regression_results.csv
exercise5_sklearn_classification_results.csv
exercise5_sklearn_clustering_results.csv
exercise5_best_k_results.csv
exercise5_sklearn_cluster_summary.csv
exercise5_knn_branch_recommendation_results.csv
exercise5_branch_recommendation_examples.csv
exercise5_keras_regression_results.csv
exercise5_keras_classification_results.csv
exercise5_keras_autoencoder_clustering_results.csv
exercise5_keras_autoencoder_cluster_summary.csv
exercise5_cross_validation_results.csv
```

These files contain model results, evaluation metrics, clustering summaries, recommendation examples, and cross-validation results.

## Installation

First, clone the repository:

```bash
git clone https://github.com/Feri-bit/my-second-project.git
cd my-second-project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

## How to Run

Run the main Python file:

```bash
python health_insurance.py
```

The program will generate datasets, train models, evaluate results, and save output CSV files in the project directory.

## Project Structure

```text
health-insurance-smart-analysis/
│
├── health_insurance.py
├── README.md
├── requirements.txt
├── LICENSE
├── examples/
│   └── sample_run.txt
└── output files generated after running the project
```

## Example Use Cases

This project can be used for:

* Learning machine learning pipelines
* Comparing regression and classification algorithms
* Understanding clustering and PCA
* Practicing TensorFlow/Keras models
* Creating a beginner-friendly educational ML project
* Studying how recommendation systems can be applied to health insurance services

## Future Improvements

Planned improvements:

* Add visual charts for model comparison
* Add a cleaner modular project structure
* Add unit tests
* Add real-world dataset support
* Add Jupyter Notebook examples
* Add a simple web dashboard
* Improve documentation and examples
* Add model saving and loading support

## Author

Farzan Asadi Shekofti

## License

This project is licensed under the MIT License.
