import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
movies = pd.read_csv("ml-100k/u.item", sep="|", names=["item_id", "title"], usecols=[0, 1], encoding="latin-1")

# Merge ratings + movie titles
data = pd.merge(ratings, movies, on="item_id")

# Create user-item matrix
user_item_matrix = data.pivot_table(index="user_id", columns="title", values="rating")

# Item-based similarity (cosine similarity between movie columns)
item_similarity = pd.DataFrame(
    cosine_similarity(user_item_matrix.T.fillna(0)),
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

def recommend(movie_title: str, top_n: int = 5):
    if movie_title not in item_similarity.columns:
        return []
    # Sort movies by similarity
    sim_scores = item_similarity[movie_title].sort_values(ascending=False)
    sim_scores = sim_scores.drop(movie_title)  # remove itself
    return sim_scores.head(top_n).index.tolist()
