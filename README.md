# Income Prediction with Fairness-Aware Machine Learning

## Overview
This project focuses on income prediction using fairness-aware machine learning models. It integrates structured census data, economic indicators, and bias detection techniques to improve accuracy and fairness in AI-driven decision-making.

## Features
- Data preprocessing and exploratory analysis of the Adult Income Dataset
- Implementation of multiple machine learning models for income prediction
- Fairness metrics evaluation and bias mitigation techniques
- Interactive web interface for model exploration and prediction

## Dataset
The project uses the Adult Income Dataset from the UCI Machine Learning Repository, which contains demographic information and income labels (>50K or <=50K) for individuals.

## System Requirements
- Python 3.8 or higher
- 8GB RAM (16GB recommended for larger experiments)
- 5GB free storage space

## Installation and Setup

### 1. Clone the Repository
```
git clone https://github.com/your-username/income-prediction-fairness.git
cd income-prediction-fairness
```

### 2. Set Up a Virtual Environment
```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Run the Web Application
```
python app.py
```
Then open your browser and navigate to http://localhost:5000

## Project Structure
- `data/`: Contains the dataset files
- `notebooks/`: Jupyter notebooks for exploratory data analysis
- `src/`: Source code for data processing, model training, and evaluation
- `static/`: Static files for the web application
- `templates/`: HTML templates for the web application
- `app.py`: Main application file for the web interface

## License
This project is licensed under the MIT License - see the LICENSE file for details.
