# Vehicle CO₂ Emissions Prediction

**Author:** Mohammad Taha — November 2025

## Project Overview

This project builds a regression model to predict vehicle CO₂ emissions from technical specifications such as engine size, fuel consumption, and vehicle class. Using data from **Transport Canada**, the final **Random Forest Regressor** reaches an **R² of 0.9972** on the test set in the saved notebook run. That figure is likely optimistic: see [Limitations of the reported scores](#limitations-of-the-reported-scores).

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
- **Duplicate handling**: 1,103 of 7,385 rows (15%) are exact duplicates. They were kept, and the random split happens afterwards, so identical rows can appear in both training and test sets
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

The notebook reports 1,103 exact duplicate rows out of 7,385 and splits the data at random after keeping them. A test row can therefore have an identical copy in the training set, which inflates the test scores above. They have not been re-measured on a de-duplicated split. CO₂ emissions are also close to a direct function of fuel consumption, so a high R² here says little about harder prediction problems.

### Key Insights

- **Exceptional predictive accuracy**: R² > 0.99 indicates CO₂ emissions are highly predictable from vehicle specifications
- **Low prediction error**: Average error under 2 g/km is excellent for practical applications
- **Strong generalization**: Test performance matches validation performance
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
