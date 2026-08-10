# 🤖 Customer Churn Prediction using ANN

A Deep Learning project that predicts whether a bank customer is likely to churn using an Artificial Neural Network (ANN).

The project includes data preprocessing, categorical encoding, feature scaling, ANN model training, TensorBoard monitoring, Early Stopping, model serialization, and a Streamlit web application for real-time churn prediction.

---

## 📌 Project Overview

Customer churn is an important problem for banks and financial institutions. Identifying customers who are likely to leave can help businesses take proactive customer-retention actions.

This project uses customer banking information such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card status
- Active Member status
- Estimated Salary

The trained ANN predicts the probability that a customer will churn.

---

## 🎯 Objective

The main objectives of this project are:

- Preprocess customer data
- Encode categorical variables
- Scale numerical features
- Build an Artificial Neural Network
- Train the model for binary classification
- Monitor training using TensorBoard
- Use Early Stopping
- Save the trained model and preprocessing objects
- Build a Streamlit web application
- Predict customer churn probability for new customers

---

## 📂 Dataset

The project uses the `Churn_Modelling.csv` dataset.

### Dataset Columns

| Column | Description |
|---|---|
| `RowNumber` | Row number |
| `CustomerId` | Unique customer ID |
| `Surname` | Customer surname |
| `CreditScore` | Customer credit score |
| `Geography` | Customer's country |
| `Gender` | Customer gender |
| `Age` | Customer age |
| `Tenure` | Number of years with the bank |
| `Balance` | Customer account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is an active member |
| `EstimatedSalary` | Estimated customer salary |
| `Exited` | Target variable indicating churn |

### Target Variable

```text
Exited = 0 → Customer did not churn
Exited = 1 → Customer churned
```

---

## 🔄 Machine Learning Pipeline

```text
Customer Dataset
       │
       ▼
Data Preprocessing
       │
       ▼
Remove Unnecessary Columns
       │
       ▼
Categorical Encoding
   ┌───┴────┐
   ▼        ▼
Gender   Geography
Label     One-Hot
Encoder   Encoder
   └───┬────┘
       ▼
Train / Test Split
       │
       ▼
StandardScaler
       │
       ▼
Artificial Neural Network
       │
       ▼
Model Training
   ┌───┴────┐
   ▼        ▼
TensorBoard  Early Stopping
       │
       ▼
Saved Model
       │
       ▼
Streamlit Application
       │
       ▼
Churn Probability
```

---

## 🧹 Data Preprocessing

### 1. Remove Unnecessary Columns

The following columns are removed:

```python
df = df.drop(
    ['RowNumber', 'CustomerId', 'Surname'],
    axis=1
)
```

These columns are identifiers and are not used as predictive features.

### 2. Encode Gender

`LabelEncoder` is used for the `Gender` column:

```python
label_encoder_gender = LabelEncoder()

df['Gender'] = label_encoder_gender.fit_transform(
    df['Gender']
)
```

The encoder is saved using Pickle so it can be reused during prediction.

### 3. Encode Geography

`OneHotEncoder` is used for the `Geography` column.

The resulting features are:

```text
Geography_France
Geography_Germany
Geography_Spain
```

### 4. Train-Test Split

The dataset is divided into:

```text
80% → Training data
20% → Testing data
```

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### 5. Feature Scaling

`StandardScaler` is used to scale the features:

```python
scalar = StandardScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
```

The fitted scaler is saved for use with new customer data.

---

## 🧠 Artificial Neural Network

The project uses a Sequential ANN.

### Architecture

```text
Input Layer
    │
    ▼
Dense Layer
64 neurons
ReLU
    │
    ▼
Dense Layer
32 neurons
ReLU
    │
    ▼
Output Layer
1 neuron
Sigmoid
```

### Model

```python
model = Sequential([
    Dense(
        64,
        activation='relu',
        input_shape=(X_train.shape[1],)
    ),
    Dense(
        32,
        activation='relu'
    ),
    Dense(
        1,
        activation='sigmoid'
    )
])
```

The model has 12 input features and 2,945 trainable parameters.

---

## ⚙️ Model Compilation

### Optimizer

```text
Adam
```

### Learning Rate

```text
0.01
```

### Loss Function

```text
Binary Crossentropy
```

### Metric

```text
Accuracy
```

```python
opt = tensorflow.keras.optimizers.Adam(
    learning_rate=0.01
)

loss = tensorflow.keras.losses.BinaryCrossentropy()

model.compile(
    optimizer=opt,
    loss=loss,
    metrics=['accuracy']
)
```

---

## 🛑 Early Stopping

Early Stopping is used to stop training when validation loss stops improving.

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

This helps avoid unnecessary training and can reduce overfitting.

---

## 📊 TensorBoard

TensorBoard is used to monitor the training process.

```python
log_dir = 'logs/fit/' + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S"
)

tensorflow_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)
```

TensorBoard can be launched with:

```python
%tensorboard --logdir logs/fit
```

---

## 🏋️ Model Training

The model is trained for up to 100 epochs with Early Stopping:

```python
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[
        tensorflow_callback,
        early_stopping
    ]
)
```

During the recorded training run, validation accuracy reached approximately 86%.

---

## 💾 Saved Model and Preprocessing Files

The trained model is saved as:

```text
model.h5
```

The preprocessing objects are saved as:

```text
scalar.pkl
label_encoder_gender.pkl
onehot_encoder_geo.pkl
```

These files allow the Streamlit application to apply the same preprocessing used during model training.

---

## 🌐 Streamlit Application

The project includes a Streamlit interface for real-time prediction.

The user can enter customer information such as:

- Geography
- Gender
- Age
- Tenure
- Credit Score
- Balance
- Number of Products
- Has Credit Card
- Active Member
- Estimated Salary

The application then performs:

```text
User Input
    ↓
Gender Encoding
    ↓
Geography One-Hot Encoding
    ↓
Feature Combination
    ↓
Standard Scaling
    ↓
ANN Model
    ↓
Churn Probability
```

### Example Result

```text
Prediction Result

✅ Customer is not likely to churn

17.08%

Churn Probability
```

The probability is displayed as a percentage.



---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Keras
- TensorBoard
- Streamlit
- Pickle

---



---

## 📌 Key Concepts Demonstrated

This project demonstrates:

- Data preprocessing
- Feature selection
- Label Encoding
- One-Hot Encoding
- Train-Test Split
- Feature Scaling
- Artificial Neural Networks
- Dense Layers
- ReLU Activation
- Sigmoid Activation
- Binary Crossentropy
- Adam Optimizer
- Early Stopping
- TensorBoard
- Model Serialization
- Streamlit Deployment
- Real-Time Prediction



---

## 🎨 Design

The original application was created in `app.py`.

For the UI redesign, I copied `app.py` and used ChatGPT to improve the interface and visual design. The redesigned version was created as:

```text
design.py
```

