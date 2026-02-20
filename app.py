"""
FitMatch - Workout Plan Recommender System
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender import (
    MostPopularRecommender,
    UserBasedCFRecommender,
    ContentBasedRecommender,
    HybridRecommender,
    split_train_test,
    evaluate_model,
)
from generate_dataset import generate_users, generate_workout_plans, generate_ratings

# ============================================================
# Page config and custom CSS
# ============================================================
st.set_page_config(page_title="FitMatch – Workout Recommender", page_icon="💪", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    .plan-card {
        background: linear-gradient(145deg, #1e1e2f, #2a2a40);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .plan-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    .plan-card h3 {
        color: #a78bfa;
        margin: 0 0 0.6rem 0;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .plan-card .meta {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-bottom: 0.5rem;
    }
    .badge {
        background: rgba(102, 126, 234, 0.15);
        color: #93a5f6;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .badge-goal {
        background: rgba(167, 139, 250, 0.15);
        color: #c4b5fd;
    }
    .plan-card .description {
        color: #9ca3af;
        font-size: 0.88rem;
        line-height: 1.45;
        margin-top: 0.4rem;
    }

    .metric-box {
        background: linear-gradient(145deg, #1e1e2f, #2a2a40);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
    }
    .metric-box h2 {
        color: #a78bfa;
        font-size: 2rem;
        margin: 0;
        font-weight: 800;
    }
    .metric-box p {
        color: #9ca3af;
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
    }

    .section-title {
        color: #e0e0e0;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }

    .demo-user-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(167, 139, 250, 0.1));
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }

    .stSelectbox > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data loading (cached)
# ============================================================
@st.cache_data
def load_data():
    """Generate or load dataset."""
    base = os.path.dirname(os.path.abspath(__file__))
    users_path = os.path.join(base, "users.csv")
    plans_path = os.path.join(base, "workout_plans.csv")
    ratings_path = os.path.join(base, "ratings.csv")

    if os.path.exists(users_path) and os.path.exists(plans_path) and os.path.exists(ratings_path):
        users = pd.read_csv(users_path)
        plans = pd.read_csv(plans_path)
        ratings = pd.read_csv(ratings_path)
    else:
        np.random.seed(42)
        users = generate_users()
        plans = generate_workout_plans()
        ratings = generate_ratings(users, plans)
        users.to_csv(users_path, index=False)
        plans.to_csv(plans_path, index=False)
        ratings.to_csv(ratings_path, index=False)

    return users, plans, ratings


@st.cache_data
def train_models(_users, _plans, _ratings):
    """Train all models and compute evaluation metrics."""
    train, test = split_train_test(_ratings)

    # Train models
    pop = MostPopularRecommender().fit(train)
    cf = UserBasedCFRecommender(k_neighbors=20).fit(train)
    cb = ContentBasedRecommender().fit(_plans, train)
    hybrid = HybridRecommender(cf_weight=0.6, cb_weight=0.4).fit(train, _plans)

    # Evaluate
    pop_metrics = evaluate_model(pop, train, test, k=5)
    cf_metrics = evaluate_model(cf, train, test, k=5)
    cb_metrics = evaluate_model(cb, train, test, k=5)
    hybrid_metrics = evaluate_model(hybrid, train, test, k=5)

    return train, test, pop, cf, cb, hybrid, pop_metrics, cf_metrics, cb_metrics, hybrid_metrics


# ============================================================
# Helper functions
# ============================================================
def render_plan_card(plan_row, rank):
    """Render a single workout plan as a styled card."""
    goal_display = plan_row['target_goal'].replace('_', ' ').title()
    st.markdown(f"""
    <div class="plan-card">
        <h3>#{rank} — {plan_row['name']}</h3>
        <div class="meta">
            <span class="badge">🏋️ {plan_row['type'].upper()}</span>
            <span class="badge">📊 {plan_row['difficulty'].title()}</span>
            <span class="badge">⏱️ {plan_row['duration_min']} min</span>
            <span class="badge badge-goal">🎯 {goal_display}</span>
        </div>
        <div class="description">{plan_row['description']}</div>
    </div>
    """, unsafe_allow_html=True)


def get_recommendations_with_details(recommender, user_id, train, plans, n=5):
    """Get recommendations and merge with plan details."""
    rec_ids = recommender.recommend(user_id, train, n=n)
    if not rec_ids:
        return pd.DataFrame()
    rec_plans = plans[plans["plan_id"].isin(rec_ids)]
    # Maintain recommendation order
    rec_plans = rec_plans.set_index("plan_id").loc[rec_ids].reset_index()
    return rec_plans


# ============================================================
# Main App
# ============================================================
def main():
    users, plans, ratings = load_data()
    train, test, pop, cf, cb, hybrid, pop_metrics, cf_metrics, cb_metrics, hybrid_metrics = train_models(users, plans, ratings)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>💪 FitMatch</h1>
        <p>AI-Powered Workout Plan Recommender System</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/dumbbell.png", width=80)
    st.sidebar.title("🔧 Configuration")

    mode = st.sidebar.radio("Selection Mode", ["By User ID", "By Profile"])

    if mode == "By User ID":
        user_id = st.sidebar.selectbox("Select User ID", sorted(users["user_id"].tolist()))
        user_row = users[users["user_id"] == user_id].iloc[0]
        st.sidebar.markdown(f"""
        **Profile:**
        - 🎂 Age: **{user_row['age']}**
        - 🏃 Fitness: **{user_row['fitness_level'].title()}**
        - 🎯 Goal: **{user_row['goal'].replace('_', ' ').title()}**
        """)
    else:
        goal = st.sidebar.selectbox("Goal", ["weight_loss", "muscle_gain", "endurance", "flexibility"],
                                     format_func=lambda x: x.replace('_', ' ').title())
        fitness = st.sidebar.selectbox("Fitness Level", ["beginner", "intermediate", "advanced"],
                                        format_func=lambda x: x.title())
        matching = users[(users["goal"] == goal) & (users["fitness_level"] == fitness)]
        if not matching.empty:
            user_id = matching.iloc[0]["user_id"]
            st.sidebar.success(f"Matched to User #{user_id}")
        else:
            user_id = users.iloc[0]["user_id"]
            st.sidebar.warning("No exact match. Showing User #1.")

    algo_choice = st.sidebar.selectbox("Algorithm", ["User-Based CF", "Content-Based", "Hybrid", "Most Popular"])
    algo_map = {
        "User-Based CF": cf,
        "Content-Based": cb,
        "Hybrid": hybrid,
        "Most Popular": pop,
    }
    selected_model = algo_map[algo_choice]

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommendations", "📊 Metrics Dashboard", "👥 Demo Panel", "📋 Dataset Info"])

    # ---- TAB 1: Recommendations ----
    with tab1:
        st.markdown(f'<div class="section-title">Top 5 Recommendations for User #{user_id} — {algo_choice}</div>', unsafe_allow_html=True)
        rec_plans = get_recommendations_with_details(selected_model, user_id, train, plans, n=5)

        if rec_plans.empty:
            st.info("No recommendations available for this user (cold-start). Try a different algorithm.")
        else:
            for i, (_, row) in enumerate(rec_plans.iterrows(), 1):
                render_plan_card(row, i)

    # ---- TAB 2: Metrics Dashboard ----
    with tab2:
        st.markdown('<div class="section-title">Performance Comparison</div>', unsafe_allow_html=True)

        metrics_df = pd.DataFrame({
            "Model": ["Most Popular (Baseline)", "User-Based CF", "Content-Based (TF-IDF)", "Hybrid (CF + CB)"],
            "Precision@5": [pop_metrics["Precision@5"], cf_metrics["Precision@5"],
                            cb_metrics["Precision@5"], hybrid_metrics["Precision@5"]],
            "Recall@5": [pop_metrics["Recall@5"], cf_metrics["Recall@5"],
                         cb_metrics["Recall@5"], hybrid_metrics["Recall@5"]],
            "Users Evaluated": [pop_metrics["Users Evaluated"], cf_metrics["Users Evaluated"],
                                cb_metrics["Users Evaluated"], hybrid_metrics["Users Evaluated"]],
        })

        # Summary metric boxes
        col1, col2, col3, col4 = st.columns(4)
        colors = ["#667eea", "#764ba2", "#f093fb", "#4facfe"]
        for i, (_, row) in enumerate(metrics_df.iterrows()):
            with [col1, col2, col3, col4][i]:
                st.markdown(f"""
                <div class="metric-box">
                    <h2>{row['Precision@5']:.3f}</h2>
                    <p>Precision@5</p>
                    <br/>
                    <h2>{row['Recall@5']:.3f}</h2>
                    <p>Recall@5</p>
                    <br/>
                    <p style="color: #667eea; font-weight: 600;">{row['Model']}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Comparison table
        st.markdown('<div class="section-title">Detailed Comparison Table</div>', unsafe_allow_html=True)
        styled_df = metrics_df.copy()
        styled_df["Precision@5"] = styled_df["Precision@5"].map("{:.4f}".format)
        styled_df["Recall@5"] = styled_df["Recall@5"].map("{:.4f}".format)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Bar chart
        st.markdown('<div class="section-title">Visual Comparison</div>', unsafe_allow_html=True)
        chart_df = metrics_df.set_index("Model")[["Precision@5", "Recall@5"]]
        st.bar_chart(chart_df, height=350)

    # ---- TAB 3: Demo Panel ----
    with tab3:
        st.markdown('<div class="section-title">Recommendations for 3 Sample Users</div>', unsafe_allow_html=True)

        # Pick 3 diverse users
        demo_users = []
        for goal in ["weight_loss", "muscle_gain", "endurance"]:
            candidates = users[users["goal"] == goal]
            if not candidates.empty:
                demo_users.append(candidates.iloc[0])

        if len(demo_users) < 3:
            demo_users = [users.iloc[i] for i in range(3)]

        for demo_user in demo_users:
            uid = demo_user["user_id"]
            st.markdown(f"""
            <div class="demo-user-header">
                <strong>👤 User #{uid}</strong> — Age: {demo_user['age']} |
                Fitness: {demo_user['fitness_level'].title()} |
                Goal: {demo_user['goal'].replace('_', ' ').title()}
            </div>
            """, unsafe_allow_html=True)

            col_cf, col_pop = st.columns(2)

            with col_cf:
                st.markdown("**🤖 User-Based CF Recommendations:**")
                cf_recs = get_recommendations_with_details(cf, uid, train, plans, n=5)
                if not cf_recs.empty:
                    for _, row in cf_recs.iterrows():
                        goal_disp = row['target_goal'].replace('_', ' ').title()
                        st.markdown(f"- **{row['name']}** — {row['type'].title()} · {row['difficulty'].title()} · {row['duration_min']}min · 🎯 {goal_disp}")
                else:
                    st.caption("No CF recommendations available.")

            with col_pop:
                st.markdown("**⭐ Most Popular Recommendations:**")
                pop_recs = get_recommendations_with_details(pop, uid, train, plans, n=5)
                if not pop_recs.empty:
                    for _, row in pop_recs.iterrows():
                        goal_disp = row['target_goal'].replace('_', ' ').title()
                        st.markdown(f"- **{row['name']}** — {row['type'].title()} · {row['difficulty'].title()} · {row['duration_min']}min · 🎯 {goal_disp}")

            st.markdown("---")

    # ---- TAB 4: Dataset Info ----
    with tab4:
        st.markdown('<div class="section-title">Dataset Statistics</div>', unsafe_allow_html=True)

        total_possible = len(users) * len(plans)
        sparsity = 1 - len(ratings) / total_possible

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-box"><h2>{len(users)}</h2><p>Users</p></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-box"><h2>{len(plans)}</h2><p>Workout Plans</p></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-box"><h2>{len(ratings)}</h2><p>Ratings</p></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-box"><h2>{sparsity:.1%}</h2><p>Sparsity</p></div>""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Fitness Level Distribution**")
            st.bar_chart(users["fitness_level"].value_counts())
        with col_b:
            st.markdown("**Goal Distribution**")
            st.bar_chart(users["goal"].value_counts())

        st.markdown('<div class="section-title">Rating Distribution</div>', unsafe_allow_html=True)
        st.bar_chart(ratings["rating"].value_counts().sort_index())

        st.markdown('<div class="section-title">Sample Data</div>', unsafe_allow_html=True)
        with st.expander("👥 Users (first 10)"):
            st.dataframe(users.head(10), use_container_width=True, hide_index=True)
        with st.expander("🏋️ Workout Plans (first 10)"):
            st.dataframe(plans[["plan_id", "name", "type", "difficulty", "duration_min", "target_goal"]].head(10),
                         use_container_width=True, hide_index=True)
        with st.expander("⭐ Ratings (first 15)"):
            st.dataframe(ratings.head(15), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
