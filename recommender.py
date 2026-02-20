"""
FitMatch - Recommender Engine
Implements User-Based CF, Content-Based, Hybrid, and Most Popular recommenders
with evaluation utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# Train / Test Split
# ============================================================
def split_train_test(ratings, test_size=0.2, seed=42):
    """Stratified-random 80/20 split ensuring every user has at least 1 train rating."""
    np.random.seed(seed)
    shuffled = ratings.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_list, test_list = [], []
    for uid, group in shuffled.groupby("user_id"):
        if len(group) <= 1:
            train_list.append(group)
        else:
            n_test = max(1, int(len(group) * test_size))
            test_list.append(group.iloc[:n_test])
            train_list.append(group.iloc[n_test:])

    train = pd.concat(train_list).reset_index(drop=True)
    test = pd.concat(test_list).reset_index(drop=True)
    return train, test


# ============================================================
# 1. Most Popular Recommender (Baseline)
# ============================================================
class MostPopularRecommender:
    """Recommend plans with the highest average rating across all users."""

    def __init__(self):
        self.popular_plans = None

    def fit(self, train_ratings):
        self.popular_plans = (
            train_ratings.groupby("plan_id")["rating"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        return self

    def recommend(self, user_id, train_ratings, n=5):
        """Return top-n most popular plans the user hasn't rated yet."""
        rated = set(train_ratings[train_ratings["user_id"] == user_id]["plan_id"])
        candidates = self.popular_plans[~self.popular_plans["plan_id"].isin(rated)]
        return candidates.head(n)["plan_id"].tolist()


# ============================================================
# 2. User-Based Collaborative Filtering
# ============================================================
class UserBasedCFRecommender:
    """User-based collaborative filtering using cosine similarity on the rating matrix."""

    def __init__(self, k_neighbors=20):
        self.k = k_neighbors
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_ids = None
        self.plan_ids = None

    def fit(self, train_ratings):
        # Build user-item matrix (users × plans), fill missing with 0
        self.user_item_matrix = train_ratings.pivot_table(
            index="user_id", columns="plan_id", values="rating", fill_value=0
        )
        self.user_ids = self.user_item_matrix.index.tolist()
        self.plan_ids = self.user_item_matrix.columns.tolist()

        # Compute cosine similarity between users
        self.similarity_matrix = pd.DataFrame(
            cosine_similarity(self.user_item_matrix),
            index=self.user_ids,
            columns=self.user_ids
        )
        return self

    def recommend(self, user_id, train_ratings, n=5):
        """Predict ratings for unseen plans and return top-n."""
        if user_id not in self.user_ids:
            return []  # Cold-start fallback

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_plans = set(user_ratings[user_ratings > 0].index)

        # Get top-k similar users (exclude self)
        sim_scores = self.similarity_matrix[user_id].drop(user_id)
        top_neighbors = sim_scores.nlargest(self.k)

        predictions = {}
        for pid in self.plan_ids:
            if pid in rated_plans:
                continue

            # Weighted average of neighbor ratings
            numerator = 0.0
            denominator = 0.0
            for neighbor_id, sim in top_neighbors.items():
                neighbor_rating = self.user_item_matrix.loc[neighbor_id, pid]
                if neighbor_rating > 0:
                    numerator += sim * neighbor_rating
                    denominator += abs(sim)

            if denominator > 0:
                predictions[pid] = numerator / denominator

        # Sort by predicted rating, return top-n
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in sorted_preds[:n]]


# ============================================================
# 3. Content-Based Recommender (TF-IDF)
# ============================================================
class ContentBasedRecommender:
    """Content-based filtering using TF-IDF on plan descriptions + cosine similarity."""

    def __init__(self):
        self.tfidf_matrix = None
        self.plan_ids = None
        self.plans_df = None
        self.similarity_matrix = None

    def fit(self, plans_df, train_ratings=None):
        self.plans_df = plans_df.copy()
        self.plan_ids = plans_df["plan_id"].tolist()

        # Combine text features for TF-IDF
        text_features = (
            plans_df["description"] + " " +
            plans_df["type"] + " " +
            plans_df["difficulty"] + " " +
            plans_df["target_goal"]
        )

        vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = vectorizer.fit_transform(text_features)

        # Plan-plan similarity
        self.similarity_matrix = pd.DataFrame(
            cosine_similarity(self.tfidf_matrix),
            index=self.plan_ids,
            columns=self.plan_ids
        )
        return self

    def recommend(self, user_id, train_ratings, n=5):
        """Recommend plans similar to what the user has rated highly."""
        user_ratings = train_ratings[train_ratings["user_id"] == user_id]
        if user_ratings.empty:
            return []

        # Get plans rated >= 4 by this user
        liked_plans = user_ratings[user_ratings["rating"] >= 4]["plan_id"].tolist()
        if not liked_plans:
            liked_plans = user_ratings.nlargest(3, "rating")["plan_id"].tolist()

        rated = set(user_ratings["plan_id"])
        scores = {}

        for pid in liked_plans:
            if pid in self.similarity_matrix.index:
                for candidate_pid, sim in self.similarity_matrix[pid].items():
                    if candidate_pid not in rated:
                        scores[candidate_pid] = scores.get(candidate_pid, 0) + sim

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in sorted_scores[:n]]


# ============================================================
# 4. Hybrid Recommender
# ============================================================
class HybridRecommender:
    """Weighted hybrid of User-Based CF and Content-Based scores."""

    def __init__(self, cf_weight=0.6, cb_weight=0.4, k_neighbors=20):
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf = UserBasedCFRecommender(k_neighbors=k_neighbors)
        self.cb = ContentBasedRecommender()

    def fit(self, train_ratings, plans_df):
        self.cf.fit(train_ratings)
        self.cb.fit(plans_df, train_ratings)
        return self

    def recommend(self, user_id, train_ratings, n=5):
        """Blend CF and CB recommendations using rank-based scoring."""
        cf_recs = self.cf.recommend(user_id, train_ratings, n=n * 2)
        cb_recs = self.cb.recommend(user_id, train_ratings, n=n * 2)

        scores = {}
        for rank, pid in enumerate(cf_recs):
            scores[pid] = scores.get(pid, 0) + self.cf_weight * (1.0 / (rank + 1))
        for rank, pid in enumerate(cb_recs):
            scores[pid] = scores.get(pid, 0) + self.cb_weight * (1.0 / (rank + 1))

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in sorted_scores[:n]]


# ============================================================
# Evaluation Metrics
# ============================================================
def precision_at_k(recommended, relevant, k=5):
    """Precision@K: fraction of recommended items that are relevant."""
    rec_set = set(recommended[:k])
    rel_set = set(relevant)
    if not rec_set:
        return 0.0
    return len(rec_set & rel_set) / len(rec_set)


def recall_at_k(recommended, relevant, k=5):
    """Recall@K: fraction of relevant items that are recommended."""
    rec_set = set(recommended[:k])
    rel_set = set(relevant)
    if not rel_set:
        return 0.0
    return len(rec_set & rel_set) / len(rel_set)


def evaluate_model(recommender, train, test, k=5, threshold=4):
    """
    Evaluate a recommender on the test set.
    Relevant items = test items with rating >= threshold.
    Returns average Precision@K and Recall@K.
    """
    precisions, recalls = [], []

    test_users = test["user_id"].unique()
    for uid in test_users:
        # Relevant items from test set
        user_test = test[test["user_id"] == uid]
        relevant = user_test[user_test["rating"] >= threshold]["plan_id"].tolist()

        if not relevant:
            continue

        # Get recommendations
        recommended = recommender.recommend(uid, train, n=k)

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))

    return {
        "Precision@5": np.mean(precisions) if precisions else 0.0,
        "Recall@5": np.mean(recalls) if recalls else 0.0,
        "Users Evaluated": len(precisions)
    }
