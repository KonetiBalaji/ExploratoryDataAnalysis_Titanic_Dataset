# Titanic EDA Project

## Exploratory Data Analysis on Titanic Dataset

### Project Overview
This project involves exploratory data analysis (EDA) of the Titanic dataset to uncover patterns influencing survival rates among passengers. It demonstrates foundational skills such as data cleaning, statistical analysis, visualization, and correlation assessment.

###  Folder Structure

```
EDA_Titanic_Project/
├── dataset/
│   └── titanic.csv
├── images/
│   ├── survival_gender.png
│   ├── survival_class.png
│   ├── age_distribution.png
│   └── correlation_heatmap.png
├── Titanic_Dataset.py
└── requirements.txt
```

###  Key Insights

- **Survival Rate by Gender:**
  - Females: **74.2%**
  - Males: **18.9%**

- **Survival Rate by Passenger Class:**
  - 1st Class: **62.9%**
  - 2nd Class: **47.3%**
  - 3rd Class: **24.2%**

- **Correlations:**
  - Strong negative correlation between **Fare** and **Passenger Class** (`-0.55`).
  - Positive correlation between **Fare** and **Survival** (`0.26`).

###  Visualizations

- **Survival Count by Gender:** Clearly demonstrates females' higher survival rate.
- **Survival Count by Passenger Class:** Highlights survival decreasing from 1st to 3rd class.
- **Age Distribution by Survival:** Provides insights into age demographics related to survival.
- **Correlation Heatmap:** Illustrates numeric correlations between features clearly.

###  Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn

###  Instructions to Run Locally

1. **Clone the repository or download the files.**
2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the Python script:**

```bash
python Titanic_Dataset.py
```

Or open the Jupyter notebook:

```bash
jupyter notebook
```

###  Conclusion

Passenger survival aboard the Titanic was significantly influenced by gender, socioeconomic status (class), and fare paid. Higher-class and female passengers had significantly better chances of survival.

---

###  Future Work

- Apply machine learning models to predict survival.
- Extend analysis to incorporate additional factors or external datasets.
