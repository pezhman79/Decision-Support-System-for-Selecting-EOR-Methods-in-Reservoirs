# Decision-Support-System-for-Selecting-EOR-Methods-in-Reservoirs

This repository contains a machine learning model built to aid in decision-making for selecting the appropriate Enhanced Oil Recovery (EOR) method in a reservoir. The dataset includes various reservoir parameters, and the goal is to predict the most suitable EOR technique based on these parameters.

## Dataset

The dataset used for this model is stored in the file `eordata.xlsx` and contains the following columns:
- **Observed**: The target variable representing the EOR method used.
- **Country**: Country code or region of the reservoir.
- **Number**: Unique identifier for each reservoir.
- Various features representing reservoir properties, including geological and production data.

## Project Structure

- `data/`: Contains the raw dataset.
- `model/`: Contains the scripts for preprocessing, training, and evaluation of machine learning models.
- `eordata.xlsx`: The dataset used for model training and testing.

## Requirements

To run this project, you need the following Python libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost
- tensorflow

You can install these libraries using pip:

```bash
pip install -r requirements.txt
git clone https://github.com/yourusername/repository-name.git
cd repository-name
```
## Load and preprocess the dataset:

```bash
import pandas as pd

df = pd.read_excel("data/eordata.xlsx")
df = df.dropna().reset_index(drop=True)

```
Train and evaluate the model using various machine learning techniques such as Random Forest, XGBoost, and Stacking Classifiers.
After training, the model can be used to predict the best EOR method based on new reservoir data.

## Model Evaluation
The model is evaluated based on classification accuracy, and results are displayed for both validation and test datasets.

## Contributing
Contributions to improve the model or add additional features are welcome! Please fork the repository and submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.




