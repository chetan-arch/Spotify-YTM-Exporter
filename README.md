# Spotify-YTM-Exporter
Export Spotify liked songs to Youtube Music along with a simple recommendation system built in

# Secrets & tokens
.env
*.env
.spotify_cache
.yt_headers.json
.yt_headers_resolved.json
oauth.json
oauth_credentials.json

# Datasets and local outputs
data/*
!data/.gitkeep

# Python/Node noise
__pycache__/
*.pyc
*.log
node_modules/

# OS/editor
.DS_Store
.vscode/


This repository contains the 3 main files. 

This project exports a user’s Spotify Liked Songs to a new YouTube Music playlist and then proposes content-based recommendations using an offline Kaggle dataset (no Spotify audio-features API required).
Spotify: Official OAuth and Web API (Spotipy).
YouTube Music: Header-based auth (cookies + UA). We parse ytcfg from https://music.youtube.com to obtain INNERTUBE_API_KEY and INNERTUBE_CONTEXT, then call youtubei endpoints (e.g., playlist/create, playlist/add_item, browse/edit_playlist).
Offline recommender: CSV(s) under data/ provide numeric features (e.g., danceability, energy). We standardize features and score candidates by cosine similarity to the centroid of the user’s liked items (matched to the dataset), with artist-level fallback.

Environment & Dependencies

Python: 3.9+ recommended.
Backend: FastAPI, Uvicorn, Spotipy, ytmusicapi, rapidfuzz, numpy, pandas, scikit-learn.
Frontend: React (Vite dev server assumed at http://127.0.0.1:5173).

Setup Checklist:
.env set with valid Spotify credentials and redirect URI.
 data/ has at least one usable CSV.
 YouTube Music headers captured from music.youtube.com (XHR request) while logged in; include full cookie and user-agent.
 Backend running: uvicorn main:app --reload (or your module name).
 Frontend running: npm run dev (or your Vite script).
 Open the frontend, connect both accounts, run export, fetch suggestions.
