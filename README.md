# Ford-car-prediction
🚗 Ford Car Price Prediction using XGBoost | Machine Learning + Streamlit Web App for predicting Ford car prices based on vehicle features.
# 🚗 Ford Car Price Prediction

A Machine Learning web application that predicts the estimated price of Ford cars based on different vehicle features. The project uses **XGBoost Regression** for prediction and **Streamlit** to provide an interactive and user-friendly web interface.

## 📌 Project Overview

This project predicts the price of a Ford car using features such as:

* 🚘 Car Model
* 📅 Manufacturing Year
* 🛣️ Mileage
* ⚙️ Transmission
* ⛽ Fuel Type
* 💷 Tax
* 📊 MPG
* 🔧 Engine Size

The trained **XGBoost Regressor** analyzes these features and provides an estimated car price through the Streamlit application.

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Streamlit**
* **Pickle**

## 🤖 Machine Learning Model

The project uses:

**XGBRegressor**

Model configuration:

* `n_estimators = 300`
* `learning_rate = 0.05`
* `max_depth = 6`
* `subsample = 0.8`
* `colsample_bytree = 0.8`
* `random_state = 42`

### 📊 Model Performance

**R² Score: 0.94**

The model achieved an R² score of approximately **94%**, indicating a strong fit on the test dataset.

## 🔄 Data Preprocessing

The dataset was preprocessed before training:

### Transmission Encoding

```python
{
    "Manual": 0,
    "Automatic": 1,
    "Semi-Auto": 2
}
```

### Fuel Type Encoding

```python
{
    "Petrol": 0,
    "Diesel": 1,
    "Hybrid": 2,
    "Electric": 3,
    "Other": 4
}
```

The `model` feature was converted using one-hot encoding:

```python
df = pd.get_dummies(
    df,
    columns=["model"],
    drop_first=True,
    dtype=int
)
```

## ✨ Features

* 🚘 Ford model selection
* 📅 Year selection
* 🛣️ Mileage input
* ⚙️ Transmission selection
* ⛽ Fuel type selection
* 💷 Tax input
* 📊 MPG input
* 🔧 Engine size input
* 🔮 Real-time price prediction
* 🖥️ Interactive Streamlit interface
* 🖼️ Ford car image display

## 📂 Project Structure

```text
ford-car-price-prediction/
│
├── app.py
├── ford_model.pkl
├── requirements.txt
├── README.md
│
└── images/
    ├── fiesta.jpg
    ├── focus.jpg
    ├── mustang.jpg
    ├── ranger.jpg
    └── ...
```

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ford-car-price-prediction.git
```

Move into the project directory:

```bash
cd ford-car-price-prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Requirements

Example `requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
xgboost
```

## 🖥️ Application Workflow

```text
User Input
    ↓
Data Preprocessing
    ↓
Feature Encoding
    ↓
XGBoost Regressor
    ↓
Predicted Ford Car Price
    ↓
Streamlit UI
```

## 🎯 Future Improvements

* Add more Ford models
* Improve prediction accuracy
* Add interactive price analysis
* Add model comparison
* Add automatic car image generation
* Deploy the application online

## 👨‍💻 Author

**Vikas Kumar**

B.Tech CSE | Machine Learning & AI Enthusiast

---

⭐ If you found this project useful, consider giving it a star!
