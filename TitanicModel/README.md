# Titanic model

This python model implements training of logistic regression model.

## Installation

1. Install Python 3
    * On Linux use `sudo apt-get install python3.7`
    * On Windows consider using [this guide](https://www.python.org/downloads/).
2. Install requirements `pip install -r requirements.txt`

## Usage

Use `train_model` function from `model.py` to read your train data in CSV format and
save model with supportive data in specified folder.

Example:
```python
train_path = "data/train.csv"
model_path = "data/model"

train_model(train_path, model_path)
```

## About model

We use Logistic regression model with liblinear solver and L1 penalty.