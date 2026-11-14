# Vehicle CO₂ Emissions Prediction

**Author:** Mohammad Taha — November 2025

## Project Overview

This project builds a regression model to predict vehicle CO₂ emissions from technical specifications such as engine size, fuel consumption, and vehicle class. Using data from **Transport Canada**, the final **Random Forest Regressor** reaches an **R² of 0.9972** on the test set in the saved notebook run. Re-evaluated without duplicate leakage and with whole vehicle models held out, it reaches **R² 0.9948 ± 0.0039 (RMSE 4.04 g/km)**: see [Re-evaluation without leakage](#re-evaluation-without-leakage).

The model provides actionable insights for environmental policy, vehicle purchasing decisions, and automotive engineering, demonstrating how machine learning can support sustainability goals.

## Environmental Context

Transportation is a major contributor to greenhouse gas emissions. Accurate prediction of vehicle CO₂ emissions enables:

- **Informed consumer choices**: Help buyers select environmentally friendly vehicles
- **Regulatory compliance**: Support emission standards enforcement
- **Automotive R&D**: Guide manufacturers in designing cleaner vehicles
- **Carbon footprint analysis**: Enable accurate environmental impact assessments
- **Policy evaluation**: Test effectiveness of emission reduction strategies

## Dataset

**Source**: Transport Canada Vehicle Emissions Database (via Kaggle)  
**Size**: 7,385 vehicles  
**Coverage**: Multiple model years and manufacturers  
**Quality**: Complete data with no missing values

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| `Make` | Categorical | Vehicle manufacturer (e.g., Toyota, Ford, BMW) |
| `Model` | Categorical | Specific vehicle model |
| `Vehicle Class` | Categorical | Body type (SUV, Sedan, Pickup, Compact, etc.) |
| `Engine Size(L)` | Numeric | Engine displacement in liters |
| `Cylinders` | Numeric | Number of engine cylinders |
| `Transmission` | Categorical | Transmission type and number of gears |
| `Fuel Type` | Categorical | Gasoline, Diesel, E85, etc. |
| `Fuel Consumption City (L/100 km)` | Numeric | City driving fuel consumption |
| `Fuel Consumption Hwy (L/100 km)` | Numeric | Highway driving fuel consumption |
| `Fuel Consumption Comb (L/100 km)` | Numeric | Combined fuel consumption |
| `Fuel Consumption Comb (mpg)` | Numeric | Combined fuel economy (miles per gallon) |
| **`CO2 Emissions(g/km)`** | **Numeric (Target)** | **Carbon dioxide emissions** |

## Technologies and Tools

- **Python 3.x**: Core programming language
- **scikit-learn**: Machine learning models and pipelines
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **KaggleHub**: Dataset integration
- **Jupyter Notebook**: Interactive development

## Methodology

### 1. Data Exploration and Cleaning

- **Completeness check**: Verified no missing values
- **Duplicate handling**: 1,103 of 7,385 rows (15%) are exact duplicates. The notebook keeps them before a random split; the re-evaluation below removes them first
- **Outlier analysis**: Identified but retained (represent real extreme cases)
- **Distribution analysis**: Examined target variable and feature relationships

### 2. Feature Engineering

Created three powerful engineered features:

- **`engine_per_cylinder`**: Engine size divided by number of cylinders
  - Captures engine design efficiency
  - Larger values indicate bigger cylinders → higher emissions

- **`city_hwy_ratio`**: City fuel consumption / Highway fuel consumption
  - Measures urban inefficiency
  - Higher values indicate vehicles poorly optimized for city driving

- **`total_consumption`**: Sum of city and highway consumption
  - Aggregate fuel usage metric

### 3. Data Preprocessing Pipeline

- **Categorical encoding**: One-hot encoding for Make, Model, Vehicle Class, Transmission, Fuel Type
- **Feature scaling**: StandardScaler for numeric features (essential for SVR)
- **Train/Validation/Test split**: 60% / 20% / 20%
- **Pipeline construction**: Integrated preprocessing and modeling

### 4. Model Training and Comparison

Four regression algorithms were evaluated:

#### Linear Regression
- Baseline model
- Surprisingly effective (R² = 0.9906)
- Indicates near-linear relationship in transformed feature space

#### Random Forest Regressor (Best Model)
- Ensemble of 100 decision trees
- Handles non-linear relationships
- Robust to outliers and feature interactions

#### Support Vector Regressor (SVR) with RBF Kernel
- Non-linear mapping via kernel trick
- Requires careful feature scaling

#### Gradient Boosting Regressor
- Sequential ensemble learning
- Strong performance but slightly behind Random Forest

## Results

### Model Performance Comparison

| Model | R² (Train) | RMSE (Train) | MAE (Train) | R² (Val) | RMSE (Val) | MAE (Val) |
|-------|------------|--------------|-------------|----------|------------|-----------|
| Linear Regression | 0.9966 | 3.43 | 1.83 | 0.9906 | 5.64 | 3.24 |
| **Random Forest** | **0.9995** | **1.32** | **0.80** | **0.9960** | **3.69** | **1.99** |
| SVR (RBF) | 0.9891 | 6.14 | 1.80 | 0.9901 | 5.82 | 3.33 |
| Gradient Boosting | 0.9975 | 2.92 | 2.29 | 0.9954 | 3.97 | 2.47 |

### Final Test Set Performance

The **Random Forest Regressor** achieved:

- **R² (Test)**: 0.9972
- **RMSE (Test)**: 3.03 g/km
- **MAE (Test)**: 1.94 g/km

### Limitations of the reported scores

The scores above come from the notebook's random split, taken after keeping 1,103 exact duplicate rows out of 7,385. In that split, 246 of the 1,477 test rows have an identical copy in the training set.

### Re-evaluation without leakage

`python evaluation/reevaluate.py "<path to CO2 Emissions_Canada.csv>"` repeats the notebook's features and model (RandomForest, 400 trees) under stricter splits. Results are saved in [evaluation/reevaluation_results.json](evaluation/reevaluation_results.json). Run on 2026-11-07 with scikit-learn 1.7.2.

| Evaluation | R² | RMSE (g/km) | MAE (g/km) |
|---|---:|---:|---:|
| Notebook split, duplicates kept (reproduced) | 0.9972 | 3.03 | 1.94 |
| Duplicates removed first, same random split | 0.9968 | 3.52 | 2.25 |
| **Duplicates removed, 5-fold GroupKFold by make and model** | **0.9948 ± 0.0039** | **4.04** | **2.11** |
| Baseline: straight line on combined fuel consumption and fuel type | 0.9902 | 5.82 | 3.24 |
| Baseline: straight line on combined fuel consumption only | 0.8401 | 23.66 | 14.49 |

**How to read this:**
- Duplicate leakage was real but small. Removing it raises the error by 16 to 33%, while R² barely moves.
- The grouped result is the honest estimate for a vehicle model the model has never seen.
- A straight line on fuel consumption and fuel type already explains 99.0% of the variance, because emissions follow from the fuel burnt. The random forest's gain over that line is RMSE 5.82 → 4.04 g/km. A high R² on this dataset therefore says little about harder prediction problems.
- **The grouped figures depend on how groups are assigned to folds.** scikit-learn 1.9 assigns groups to `GroupKFold` folds differently from 1.7. With scikit-learn 1.9.1 (2026-11-14), the same code gives R² 0.9937 ± 0.0064, RMSE 4.15 ± 2.03 g/km and MAE 2.11, and the fuel-and-type baseline gives RMSE 5.84. scikit-learn 1.7.2 reproduces the table above exactly. On Kaggle (scikit-learn 1.6.1, 2026-11-14) the forest gives R² 0.9945 ± 0.0052 and RMSE 4.06. The conclusion is unchanged: about R² 0.994 and RMSE 4 g/km on unseen vehicle models, roughly 1.7 g/km better than the straight-line baseline. The large RMSE spread comes from one hard fold.

### Kaggle-ready notebook

[evaluation/leakage_free_evaluation.ipynb](evaluation/leakage_free_evaluation.ipynb) runs the same evaluation as a notebook, with saved outputs (scikit-learn 1.9.1). It finds the CSV under `/kaggle/input`, in `CO2_CSV`, or downloads it with kagglehub.

### Key Insights

- **Predictable target**: R² > 0.99, but a straight line on fuel consumption and fuel type already reaches 0.990
- **Error on unseen vehicle models**: RMSE 4.04 g/km and MAE 2.11 g/km in grouped cross-validation (the notebook's 1.94 g/km MAE comes from a split with duplicate leakage)
- **Generalization**: holds across held-out vehicle models (per-fold R² from 0.9872 to 0.9977)
- **Feature importance**: Combined fuel consumption is the dominant predictor, followed by engine design metrics

## Feature Importance Analysis

The Random Forest model identified the most influential predictors:

1. **Fuel Consumption Comb (L/100 km)** — Dominant factor (directly proportional to CO₂)
2. **engine_per_cylinder** — Engine design efficiency metric
3. **city_hwy_ratio** — Urban driving inefficiency indicator
4. **Vehicle Class** — Body type and weight category
5. **Fuel Type** — Different fuels have different carbon content per liter

These findings align perfectly with **physical and engineering principles**, increasing confidence in the model's reliability.

## Project Structure

```
vehicle-co2-emissions-prediction/
├── vehicle_co2_emissions_prediction.ipynb  # Main analysis notebook
├── README.md                               # This file
├── LICENSE                                 # MIT License
├── requirements.txt                        # Python dependencies
└── .gitignore                              # Git ignore rules
```

## How to Run

### Prerequisites

- Python 3.8+
- Minimum 4GB RAM

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tahamohmadf19-dev/vehicle-co2-emissions-prediction.git
   cd vehicle-co2-emissions-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook vehicle_co2_emissions_prediction.ipynb
   ```

### Running on Google Colab

1. Upload the notebook to Google Colab
2. The dataset will be automatically downloaded via KaggleHub
3. Run all cells sequentially

## Real-World Applications

### For Consumers
- Compare emissions of different vehicles before purchase
- Estimate environmental impact of driving habits
- Calculate carbon footprint for personal sustainability goals

### For Policymakers
- Evaluate effectiveness of emission standards
- Identify vehicle categories requiring stricter regulations
- Design incentive programs for low-emission vehicles

### For Manufacturers
- Benchmark new designs against market standards
- Optimize engine and transmission configurations
- Predict regulatory compliance before production

## Future Enhancements

- **Temporal analysis**: Incorporate model year trends to track emission improvements over time
- **Electric vehicle integration**: Extend model to handle EVs and plug-in hybrids
- **Uncertainty quantification**: Implement prediction intervals using quantile regression
- **Interactive web application**: Deploy as a user-friendly emissions calculator
- **API development**: Create REST API for integration with automotive databases
- **Multi-pollutant modeling**: Extend to NOx, particulate matter, and other emissions
- **Causal inference**: Use causal models to identify design changes that reduce emissions

## Educational Value

This project demonstrates:

- End-to-end regression modeling workflow
- Feature engineering for domain-specific problems
- Model selection and hyperparameter tuning
- Interpretability and validation of ML models
- Application of machine learning to environmental challenges
- Best practices in data science project documentation

## Limitations

- **Geographic scope**: Data represents Canadian market; may not generalize to other regions
- **Temporal scope**: Does not account for technological evolution over time
- **Correlation vs causation**: Model captures empirical patterns, not causal mechanisms
- **Real-world driving**: Test cycle data may differ from actual on-road emissions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Transport Canada Vehicle Emissions Database
- Kaggle Dataset: [CO2 Emission by Vehicles](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)
- scikit-learn Documentation
- IPCC Guidelines for Greenhouse Gas Inventories

## Contact

**Mohammad Taha**  
📧 Email: tahamohmadf19@gmail.com  
🔗 LinkedIn: [Mohmad Taha Alhmad](https://www.linkedin.com/in/mohmad-taha-alhmad)  
🐙 GitHub: [@tahamohmadf19-dev](https://github.com/tahamohmadf19-dev)

---

*This project contributes to environmental sustainability by making vehicle emissions transparent and predictable through data science.*
