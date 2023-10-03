# Fraud-Transaction-Detection-TechnoHack
In this, we tackle the critical problem of identifying fraudulent transactions detection using machine learning. Fraudulent transactions can have a significant impact on both individuals and financial institutions, making accurate detection essential.

Overview In this project, we perform the following steps:

Data Loading: We start by loading the credit card transaction dataset from a CSV file (tested.csv) using the pandas library. This dataset contains transaction details, including transaction amount and class labels (fraudulent or not).

Data Preprocessing: Data preprocessing is vital for any machine learning task. We apply standardization to the transaction amount using StandardScaler. Additionally, we address the class imbalance issue by oversampling the minority class (fraudulent transactions) using the RandomOverSampler from the imblearn library.

Model Selection: For fraud detection, we choose the Random Forest Classifier. Random Forest is an ensemble learning method known for its ability to handle imbalanced datasets and make accurate predictions.

Model Training: We train the Random Forest model on the preprocessed data, using simplified hyperparameters for ease of implementation.

Model Evaluation: We evaluate the model's performance using various metrics, including accuracy, the classification report, and the confusion matrix. These metrics provide insights into how well the model can distinguish between fraudulent and legitimate transactions.

Visualization: Visualizations play a crucial role in understanding both the data and the model's performance. We create visualizations for feature importance, ROC curves, precision-recall curves, histograms of features, and box plots to gain deeper insights.

Requirements

Before running the code in this repository, ensure that you have the necessary Python libraries installed. You can install them using pip: bash Copy code pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn Usage

Clone this repository: bash Copy code git clone https://github.com/your-username/Fraud-Transaction-Detection.git

Navigate to the project directory: bash Copy code cd credit-card-fraud-detection Ensure you have the tested.csv file in the same directory as the code.

Run the Jupyter Notebook or Python script to execute the project.
