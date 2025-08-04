
````markdown
# 🔧 Predictive Maintenance using LSTM – Capstone Project

**What if you could know ahead of time that a machine was going to fail — and stop it before it does?**  
This project demonstrates exactly that by using LSTM-based deep learning to predict the Remaining Useful Life (RUL) of machines based on time-series sensor data.

---

## 📁 Folder Structure

```bash
.
├── CMAPSSData/                       # Raw dataset (NASA CMAPSS)
├── rul_deployment/                  # Deployment folder for app
│   ├── app.py                       # Streamlit app
│   ├── lstm_rul_model.h5            # Trained LSTM model
│   ├── requirements.txt             # Python dependencies
│   ├── runtime.txt                  # Runtime environment for deployment
│   ├── test_FD001.txt               # Sample test file (FD001)
│   ├── utils.py                     # Helper functions
├── LICENSE                          # MIT License
├── README.md                        # Project documentation
├── RUL_for_predictive_maintenance.ipynb  # Jupyter notebook (EDA, training)
````

---

## 💡 Project Description

This capstone project predicts the **Remaining Useful Life (RUL)** of jet engines using sensor data and LSTM neural networks. The model helps forecast when maintenance should be performed before actual machine failure occurs.

We used the **CMAPSS dataset from NASA**, which provides sensor measurements and operational settings for aircraft engines under varying conditions and fault modes.

---

## 🔬 Model Overview

* **Model**: Stacked LSTM with Dense output layer
* **Input**: Time-windowed sequences of sensor data
* **Output**: Remaining Useful Life (RUL) in cycles
* **Loss Function**: Mean Squared Error (MSE)
* **Framework**: TensorFlow/Keras

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/olalekanali/Hamoye-Capstone_LSTM.git
cd predictive-maintenance-lstm
```

### 2. Install Dependencies

```bash
pip install -r rul_deployment/requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run rul_deployment/app.py
```

---

## 📊 Dataset

* Dataset: **CMAPSS** (Commercial Modular Aero-Propulsion System Simulation)
* Download from: [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

Make sure the downloaded data is placed in the `/CMAPSSData/` directory.

---

## 📈 Results

| Metric | Value     |
| ------ | --------- |
| MAE    | XX cycles |
| RMSE   | XX cycles |

---

## 🧪 Notebook

The training and evaluation process is detailed in `RUL_for_predictive_maintenance.ipynb`, including:

* Data preprocessing
* Sequence generation
* Model training and validation
* Visualization of RUL predictions

---

## ✈️ Real-World Relevance

Failures in aircraft engines, like those seen in tragic incidents such as **Flight AI171**, could potentially have been detected early using predictive maintenance models like this — making a strong case for integrating AI-driven health monitoring in safety-critical systems.

---


## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
