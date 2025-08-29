# Movie Recommendation API

A FastAPI-based movie recommendation system using collaborative filtering and content-based filtering.

## Features

- Hybrid recommendation system (collaborative + content-based)
- MovieLens 100K dataset integration
- TMDB poster integration
- RESTful API endpoints
- Interactive web interface

## API Endpoints

- `GET /` - Health check
- `GET /stats` - Dataset statistics
- `GET /popular` - Popular movies
- `GET /recommend?movie=<title>` - Get recommendations
- `GET /suggest?q=<query>` - Movie suggestions

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variable:
```bash
export TMDB_API_KEY=your_api_key_here
```

3. Run the server:
```bash
uvicorn ml_model.main:app --reload
```

4. Open http://localhost:8000

## Deployment

### Render
1. Push code to GitHub
2. Create new Web Service on Render
3. Connect your repository
4. Set environment variable: `TMDB_API_KEY`
5. Deploy

### Railway
1. Push code to GitHub
2. Create new project on Railway
3. Deploy from GitHub
4. Set environment variable: `TMDB_API_KEY`

## Environment Variables

- `TMDB_API_KEY`: Your TMDB API key for movie posters (optional)

## Project Structure

```
movie_recommendation/
├── ml_model/
│   ├── main.py              # FastAPI app
│   ├── recommender.py       # ML models
│   ├── ml-100k/            # MovieLens dataset
│   └── static/             # Frontend files
├── requirements.txt         # Python dependencies
├── render.yaml             # Render configuration
├── Procfile               # Railway/Heroku configuration
└── README.md              # This file
```
