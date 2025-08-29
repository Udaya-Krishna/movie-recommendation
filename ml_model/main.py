from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os 
import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY not found in environment variables")

app = FastAPI()

# Allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MovieRecommendationSystem:
    def __init__(self):
        self.movies = None
        self.ratings = None
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.content_similarity_matrix = None
        self.movie_features = None
        self.load_data()
        self.build_models()
    
    def load_data(self):
        """Load and preprocess MovieLens data"""
        try:
            # Load movies data
            self.movies = pd.read_csv(
                "ml-100k/u.item", sep="|", encoding="latin-1",
                names=["movie_id", "title", "release_date", "video_release", "imdb_url"] + 
                      [f"genre_{i}" for i in range(19)],
                usecols=range(24)
            )
            
            # Load ratings data
            self.ratings = pd.read_csv(
                "ml-100k/u.data", sep="\t",
                names=["user_id", "movie_id", "rating", "timestamp"]
            )
            
            # Clean movie titles (remove year from title for better matching)
            self.movies['clean_title'] = self.movies['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
            
            print(f"Loaded {len(self.movies)} movies and {len(self.ratings)} ratings")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"MovieLens data not found: {e}")
    
    def build_models(self):
        """Build collaborative filtering and content-based models"""
        print("Building recommendation models...")
        
        # 1. Build user-item matrix for collaborative filtering
        self.user_item_matrix = self.ratings.pivot_table(
            index='user_id', columns='movie_id', values='rating', fill_value=0
        ).values
        
        # 2. Build item-item similarity matrix (collaborative filtering)
        # Transpose to get item-user matrix
        item_user_matrix = self.user_item_matrix.T
        
        # Calculate cosine similarity between items
        self.item_similarity_matrix = cosine_similarity(item_user_matrix)
        
        # 3. Build content-based similarity matrix
        # Create genre features
        genre_cols = [col for col in self.movies.columns if col.startswith('genre_')]
        genre_matrix = self.movies[genre_cols].values
        
        # Calculate content similarity based on genres
        self.content_similarity_matrix = cosine_similarity(genre_matrix)
        
        print("Models built successfully!")
    
    def get_movie_id_by_title(self, title):
        """Find movie ID by title (fuzzy matching)"""
        title_lower = title.lower()
        
        # Try exact match first
        exact_match = self.movies[self.movies['title'].str.lower() == title_lower]
        if not exact_match.empty:
            return exact_match.iloc[0]['movie_id']
        
        # Try partial match with clean title
        partial_match = self.movies[
            self.movies['clean_title'].str.lower().str.contains(title_lower, na=False)
        ]
        if not partial_match.empty:
            return partial_match.iloc[0]['movie_id']
        
        # Try partial match with original title
        partial_match = self.movies[
            self.movies['title'].str.lower().str.contains(title_lower, na=False)
        ]
        if not partial_match.empty:
            return partial_match.iloc[0]['movie_id']
        
        return None
    
    def collaborative_filtering_recommend(self, movie_id, n_recommendations=10):
        """Item-based collaborative filtering"""
        if movie_id not in range(1, len(self.item_similarity_matrix) + 1):
            return []
        
        movie_idx = movie_id - 1  # Convert to 0-based index
        
        # Get similarity scores for this movie
        similarity_scores = self.item_similarity_matrix[movie_idx]
        
        # Get top similar movies (excluding the movie itself)
        similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations+1]
        similar_movie_ids = [idx + 1 for idx in similar_indices]  # Convert back to 1-based
        
        # Filter out movies that don't exist in our dataset
        valid_movie_ids = []
        for mid in similar_movie_ids:
            if mid in self.movies['movie_id'].values:
                valid_movie_ids.append(mid)
        
        return valid_movie_ids[:n_recommendations]
    
    def content_based_recommend(self, movie_id, n_recommendations=10):
        """Content-based filtering using genre similarity"""
        if movie_id not in self.movies['movie_id'].values:
            return []
        
        movie_idx = self.movies[self.movies['movie_id'] == movie_id].index[0]
        
        # Get similarity scores
        similarity_scores = self.content_similarity_matrix[movie_idx]
        
        # Get top similar movies
        similar_indices = similarity_scores.argsort()[::-1][1:n_recommendations+1]
        similar_movie_ids = self.movies.iloc[similar_indices]['movie_id'].tolist()
        
        return similar_movie_ids
    
    def hybrid_recommend(self, movie_title, n_recommendations=5):
        """Hybrid recommendation combining collaborative and content-based"""
        movie_id = self.get_movie_id_by_title(movie_title)
        
        if movie_id is None:
            return []
        
        # Get recommendations from both methods
        collab_recs = self.collaborative_filtering_recommend(movie_id, n_recommendations * 2)
        content_recs = self.content_based_recommend(movie_id, n_recommendations * 2)
        
        # Combine and weight the recommendations
        # Give more weight to collaborative filtering if the movie has enough ratings
        movie_rating_count = len(self.ratings[self.ratings['movie_id'] == movie_id])
        
        if movie_rating_count >= 10:  # Enough ratings for collaborative filtering
            # 70% collaborative, 30% content-based
            final_recs = collab_recs[:int(n_recommendations * 0.7)] + content_recs[:int(n_recommendations * 0.3)]
        else:
            # 30% collaborative, 70% content-based
            final_recs = content_recs[:int(n_recommendations * 0.7)] + collab_recs[:int(n_recommendations * 0.3)]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recs = []
        for rec in final_recs:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)
        
        return unique_recs[:n_recommendations]
    
    def get_movie_details(self, movie_ids):
        """Get movie details for given movie IDs"""
        movie_details = []
        for movie_id in movie_ids:
            movie_info = self.movies[self.movies['movie_id'] == movie_id]
            if not movie_info.empty:
                movie_details.append({
                    'movie_id': movie_id,
                    'title': movie_info.iloc[0]['title'],
                    'clean_title': movie_info.iloc[0]['clean_title']
                })
        return movie_details
    
    def get_popular_movies(self, n_movies=10):
        """Get popular movies based on rating count and average rating"""
        movie_stats = self.ratings.groupby('movie_id').agg({
            'rating': ['count', 'mean']
        }).round(2)
        
        movie_stats.columns = ['rating_count', 'avg_rating']
        movie_stats = movie_stats.reset_index()
        
        # Filter movies with at least 50 ratings
        popular_movies = movie_stats[movie_stats['rating_count'] >= 50]
        
        # Sort by average rating (descending)
        popular_movies = popular_movies.sort_values('avg_rating', ascending=False)
        
        top_movie_ids = popular_movies.head(n_movies)['movie_id'].tolist()
        return self.get_movie_details(top_movie_ids)


# Initialize the recommendation system
print("Initializing recommendation system...")
rec_system = MovieRecommendationSystem()
print("Recommendation system ready!")


def get_poster_from_tmdb(movie_name: str):
    """Fetch poster from TMDB API"""
    url = f"https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_name}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.RequestException as e:
        print(f"Error fetching poster for {movie_name}: {e}")
    
    return None


@app.get("/recommend")
def recommend(movie: str = Query(..., description="Movie name")):
    """Get ML-based movie recommendations"""
    try:
        # Get recommendations using hybrid approach
        recommended_movie_ids = rec_system.hybrid_recommend(movie, n_recommendations=5)
        
        if not recommended_movie_ids:
            raise HTTPException(status_code=404, detail="Movie not found or no recommendations available")
        
        # Get movie details
        movie_details = rec_system.get_movie_details(recommended_movie_ids)
        
        # Fetch posters
        recommendations_with_posters = []
        for movie_detail in movie_details:
            poster_url = get_poster_from_tmdb(movie_detail['clean_title'])
            recommendations_with_posters.append({
                "title": movie_detail['title'],
                "poster": poster_url or ""
            })
        
        return {
            "movie": movie,
            "recommendations": recommendations_with_posters,
            "method": "hybrid_ml"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.get("/suggest")
def suggest_movies(q: str = Query("", description="Search query")):
    """Get movie suggestions for search"""
    if not q:
        # Return popular movies if no query
        popular_movies = rec_system.get_popular_movies(10)
        return {"suggestions": [movie['title'] for movie in popular_movies]}
    
    q_lower = q.lower()
    suggestions = []
    
    for _, movie in rec_system.movies.iterrows():
        if q_lower in movie['title'].lower() or q_lower in movie['clean_title'].lower():
            suggestions.append(movie['title'])
        if len(suggestions) >= 10:
            break
    
    return {"suggestions": suggestions}


@app.get("/popular")
def get_popular():
    """Get popular movies"""
    popular_movies = rec_system.get_popular_movies(20)
    
    movies_with_posters = []
    for movie in popular_movies:
        poster_url = get_poster_from_tmdb(movie['clean_title'])
        movies_with_posters.append({
            "title": movie['title'],
            "poster": poster_url or ""
        })
    
    return {"popular_movies": movies_with_posters}


@app.get("/stats")
def get_stats():
    """Get dataset statistics"""
    return {
        "total_movies": len(rec_system.movies),
        "total_ratings": len(rec_system.ratings),
        "total_users": rec_system.ratings['user_id'].nunique(),
        "avg_rating": round(rec_system.ratings['rating'].mean(), 2),
        "rating_distribution": rec_system.ratings['rating'].value_counts().sort_index().to_dict()
    }


@app.get("/")
def health_check():
    return {
        "status": "OK", 
        "message": "ML-based Movie Recommendation API is running",
        "model": "Hybrid (Collaborative Filtering + Content-based)"
    }