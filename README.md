# 💪 FitMatch — Workout Plan Recommender System

A personalized **workout plan recommender system** that suggests the best workout plans based on user fitness profiles and preferences. Built with Python, scikit-learn, and Streamlit.

---

## 🎯 Overview

FitMatch implements **four recommendation algorithms** and compares their performance:

| # | Algorithm | Approach |
|---|-----------|----------|
| 1 | **Most Popular** (Baseline) | Recommends plans with the highest average rating |
| 2 | **User-Based Collaborative Filtering** | Cosine similarity on the user-item rating matrix |
| 3 | **Content-Based Filtering (TF-IDF)** | Cosine similarity on workout plan descriptions |
| 4 | **Hybrid (CF + Content-Based)** | Weighted blend of approaches 2 & 3 |

---

## 📊 Dataset

The dataset is **simulated** using NumPy/Pandas with a fixed random seed (`42`) for full reproducibility:

- **100 users** — with age, fitness level, and goal attributes
- **60 workout plans** — with type, difficulty, duration, target goal, and description
- **800+ ratings** — user–plan ratings on a 1–5 scale (~86% sparsity)

Files: `users.csv`, `workout_plans.csv`, `ratings.csv`
*(Auto-generated when you run `generate_dataset.py` or the notebook)*

---

## 🛠️ Tech Stack

- **Python 3.8+**
- NumPy & Pandas
- scikit-learn
- Streamlit
- Matplotlib
- Jupyter Notebook

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MaDhuManodya/RS.git
cd RS
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset

```bash
python generate_dataset.py
```

### 4. Run the Jupyter Notebook

```bash
jupyter notebook FitMatch_Recommender.ipynb
```

- Run all cells sequentially
- Covers all 5 steps: Problem Definition, Data Preparation, Model Development, Evaluation, and Demonstration

### 5. Run the Streamlit Web App

```bash
streamlit run app.py
```

- Opens in your browser at `http://localhost:8501`
- Use the sidebar to select a user ID or filter by profile
- View recommendations, metrics dashboard, and demo panel

---

## 📈 Expected Output

### Notebook

- Dataset statistics (100 users, 60 plans, 800+ ratings, ~86% sparsity)
- Distribution charts for ratings, fitness levels, goals, and plan types
- Model training output with similarity examples
- Evaluation comparison table (Precision@5 and Recall@5 for all models)
- Detailed recommendations for 3 demo users with explanations

### Streamlit App

| Tab | Description |
|-----|-------------|
| 🏋️ Recommendations | Top 5 recommended workout plans for selected user |
| 📊 Metrics | Performance metrics dashboard with comparison chart |
| 🎯 Demo | Recommendations for 3 sample users |
| 📋 Dataset | Dataset statistics and exploration |

---

## 📁 Project Structure

```
RS/
├── generate_dataset.py        # Dataset generation module
├── recommender.py             # Recommender algorithms + evaluation
├── FitMatch_Recommender.ipynb # Jupyter notebook (all 5 steps)
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── users.csv                  # Generated user data
├── workout_plans.csv          # Generated workout plan data
└── ratings.csv                # Generated ratings data
```

---

## 📝 License

This project is for educational purposes.
