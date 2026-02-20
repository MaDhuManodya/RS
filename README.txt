================================================================================
  FitMatch — Workout Plan Recommender System
  README
================================================================================

DESCRIPTION
-----------
FitMatch is a workout plan recommender system that suggests personalized
workout plans to users based on their fitness profile and preferences.
It implements three recommendation algorithms:

  1. Most Popular (Baseline) — recommends plans with highest avg rating
  2. User-Based Collaborative Filtering — cosine similarity on rating matrix
  3. Content-Based Filtering (TF-IDF) — cosine similarity on plan descriptions
  4. Hybrid (CF + Content-Based) — weighted blend of approaches 2 and 3

DATASET
-------
The dataset is simulated and generated using Python (numpy/pandas) with
a random seed (42) for full reproducibility:
  - 100 users with age, fitness level, and goal attributes
  - 60 workout plans with type, difficulty, duration, target goal, description
  - 800+ user-plan ratings on a 1–5 scale

Files: users.csv, workout_plans.csv, ratings.csv
(Auto-generated when you run generate_dataset.py or the notebook)

LIBRARIES
---------
  Python 3.8+
  numpy
  pandas
  scikit-learn
  streamlit
  matplotlib
  jupyter

Install all:
  pip install -r requirements.txt

HOW TO RUN
----------

1. Generate the dataset:
     cd RS
     python generate_dataset.py

2. Run the Jupyter Notebook:
     jupyter notebook FitMatch_Recommender.ipynb

   - Run all cells sequentially
   - The notebook covers all 5 steps: Problem Definition, Data Preparation,
     Model Development, Evaluation, and Demonstration

3. Run the Streamlit Web App:
     streamlit run app.py

   - Opens in browser at http://localhost:8501
   - Use the sidebar to select a user ID or filter by profile
   - View recommendations, metrics dashboard, and demo panel

EXPECTED OUTPUT
---------------
Notebook:
  - Dataset statistics (100 users, 60 plans, 800+ ratings, ~86% sparsity)
  - Distribution charts for ratings, fitness levels, goals, plan types
  - Model training output with similarity examples
  - Evaluation comparison table (Precision@5 and Recall@5 for all models)
  - Detailed recommendations for 3 demo users with explanations

Streamlit App:
  - Tab 1: Top 5 recommended workout plans for selected user
  - Tab 2: Performance metrics dashboard with comparison chart
  - Tab 3: Demo panel showing recommendations for 3 sample users
  - Tab 4: Dataset statistics and exploration

PROJECT STRUCTURE
-----------------
  RS/
  ├── generate_dataset.py       # Dataset generation module
  ├── recommender.py            # Recommender algorithms + evaluation
  ├── FitMatch_Recommender.ipynb # Jupyter notebook (all 5 steps)
  ├── app.py                    # Streamlit web application
  ├── requirements.txt          # Python dependencies
  ├── README.txt                # This file
  ├── users.csv                 # Generated user data
  ├── workout_plans.csv         # Generated workout plan data
  └── ratings.csv               # Generated ratings data

================================================================================
