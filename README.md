# Titanic Task

This is a test assignment for JetBrains Internship Program.
It implements a survival prediction model for the Titanic dataset.  

## Installation

1. Download repository: `https://github.com/AndreyBychkov/TitanicTask.git`
2. Run: `gradle build`
3. Follow instructions in [Python module README](TitanicModel/README.md).

## Usage

We store data in [data](data) folder.
 There you can find `train.csv` and `test.csv` files.
 
Use `TitanicModel` module to train the model.
After training model with supplementary files is stored in [data/model](data/model) directory.

Run `main.kt` to use out model for making `submission.csv`.