import json
import os
import re
import time
import math
import hashlib
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import RedirectResponse

import requests
from json import JSONDecodeError

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy import SpotifyException

from ytmusicapi import YTMusic
from rapidfuzz import fuzz

# ML bits (kept minimal; old ML endpoints removed)
import numpy as np
from scipy import sparse  # noqa: F401  (left for minimal change)
from sklearn.preprocessing import StandardScaler
from lightfm import LightFM  # noqa: F401 (left for minimal change)

import pandas as pd

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")
SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
DATA_DIR = os.getenv("DATA_DIR", "data")  # put your Kaggle CSVs here

# Derive frontend base from redirect URI unless explicitly set
_frontend_from_redirect = ""
if SPOTIFY_REDIRECT_URI:
    p = urlparse(SPOTIFY_REDIRECT_URI)
    _frontend_from_redirect = f"{p.scheme}://{p.netloc}"
FRONTEND_BASE = os.getenv("FRONTEND_BASE", _frontend_from_redirect or "http://127.0.0.1:5173")

# ----------------------------
# In-memory demo state (replace with DB in prod)
# ----------------------------
STATE: Dict[str, Any] = {
    "spotify_token": None,
    "yt_headers": None,  # dict (raw headers captured from the browser)
}

SCOPES = [
    "user-library-read",
    "user-top-read",
    "playlist-modify-private",
    "playlist-modify-public",
]

app = FastAPI(title="Spotify → YouTube Music Exporter + Recommender (Offline Dataset)")

# Open CORS fully for local dev. (Tighten later.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # "*" requires False
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------

class YTHeadersPayload(BaseModel):
    headers_json: Dict[str, Any]

class ExportRequest(BaseModel):
    playlist_name: Optional[str] = None
    also_like_on_ytm: bool = False

class ExportResult(BaseModel):
    created_playlist_id: str
    total_spotify_tracks: int
    matched: int
    added: int
    liked: int
    unmatched_samples: List[Dict[str, Any]]

class RecsRequest(BaseModel):
    count: int = 30

# ---------- Helpers (general) ----------

def _freshen_headers_for_post(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Clone headers and refresh Authorization (SAPISIDHASH ts) and basic JSON headers.
    Must be called before every POST to YT endpoints.
    """
    h = dict(headers)
    cookie = h.get("Cookie") or h.get("cookie") or ""
    origin = h.get("Origin") or "https://music.youtube.com"
    sap = _compute_sapisidhash(cookie, origin)
    if sap:
        h["Authorization"] = sap
    h["Content-Type"] = "application/json"
    h["Accept"] = "application/json"
    h.setdefault("Referer", "https://music.youtube.com/")
    h.setdefault("Accept-Language", "en-US,en;q=0.9")
    return h


def _is_valid_sp_id(s: Optional[str]) -> bool:
    return isinstance(s, str) and len(s) == 22 and s.isalnum()

def _extract_cookie_value(cookie_str: str, name: str) -> Optional[str]:
    try:
        parts = [p.strip() for p in cookie_str.split(";")]
        for p in parts:
            if not p or "=" not in p:
                continue
            k, v = p.split("=", 1)
            if k.strip() == name:
                return v.strip()
    except Exception:
        pass
    return None

def _compute_sapisidhash(cookie_str: str, origin: str) -> Optional[str]:
    """
    Build 'Authorization: SAPISIDHASH <ts>_<sha1>' using SAPISID (or __Secure-3PAPISID/__Secure-3PSID).
    sha1(f"{ts} {sapisid} {origin}")
    """
    sapisid = (
        _extract_cookie_value(cookie_str, "SAPISID")
        or _extract_cookie_value(cookie_str, "__Secure-3PAPISID")
        or _extract_cookie_value(cookie_str, "__Secure-3PSID")
    )
    if not sapisid:
        return None
    ts = str(int(time.time()))
    to_hash = f"{ts} {sapisid} {origin}".encode("utf-8")
    digest = hashlib.sha1(to_hash).hexdigest()
    return f"SAPISIDHASH {ts}_{digest}"

def _check_spotify_env():
    if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET and SPOTIFY_REDIRECT_URI):
        raise HTTPException(500, "Spotify credentials or redirect URI not configured")

def get_spotify() -> spotipy.Spotify:
    _check_spotify_env()
    oauth = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=" ".join(SCOPES),
        cache_path=".spotify_cache",
        show_dialog=False,
    )
    token_info = oauth.get_cached_token()
    if token_info:
        STATE["spotify_token"] = token_info

    if STATE["spotify_token"] is None:
        raise HTTPException(401, "Not connected to Spotify")

    if oauth.is_token_expired(STATE["spotify_token"]):
        STATE["spotify_token"] = oauth.refresh_access_token(STATE["spotify_token"]["refresh_token"])

    return spotipy.Spotify(auth=STATE["spotify_token"]["access_token"])

def get_yt() -> YTMusic:
    # Stay in header-auth mode (block OAuth confusion)
    if os.path.exists("oauth.json") or os.path.exists("oauth_credentials.json"):
        raise HTTPException(
            500,
            "Found oauth.json/oauth_credentials.json. Delete them to use header auth (or finish OAuth setup)."
        )

    # Load headers from memory or disk
    if STATE["yt_headers"] is None and os.path.exists(".yt_headers.json"):
        try:
            with open(".yt_headers.json", "r", encoding="utf-8") as f:
                STATE["yt_headers"] = json.load(f)
        except Exception:
            pass
    if STATE["yt_headers"] is None:
        raise HTTPException(401, "Not connected to YouTube Music (no headers set)")

    raw = STATE["yt_headers"]
    if not isinstance(raw, dict):
        raise HTTPException(400, "YouTube headers must be a JSON object (key/value).")

    # Normalize keys and coerce ALL values to strings
    lower = {}
    for k, v in raw.items():
        kk = str(k).lower() if isinstance(k, str) else k
        vv = "" if v is None else (v if isinstance(v, str) else str(v))
        lower[kk] = vv

    cookie = lower.get("cookie", "").strip()
    ua     = lower.get("user-agent", "").strip()
    xorig  = lower.get("x-origin", "https://music.youtube.com").strip()
    xauth  = lower.get("x-goog-authuser", "0").strip()
    authz  = lower.get("authorization", "").strip()

    if not cookie or not ua:
        raise HTTPException(400, "YouTube headers must include non-empty 'cookie' and 'user-agent'.")

    origin = "https://music.youtube.com"

    # If Authorization not provided, compute SAPISIDHASH from cookie
    if not authz:
        maybe = _compute_sapisidhash(cookie, origin)
        if maybe:
            authz = maybe

    resolved = {
        "Cookie": cookie,
        "User-Agent": ua,
        "Origin": origin,
        "x-origin": xorig or origin,
        "x-goog-authuser": xauth or "0",
        "Referer": "https://music.youtube.com/",
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if authz:
        resolved["Authorization"] = authz

    # Persist what we’ll use (debuggable)
    try:
        with open(".yt_headers_resolved.json", "w", encoding="utf-8") as f:
            json.dump(resolved, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Pass dict directly; ytmusicapi accepts header dicts
    return YTMusic(auth=resolved)

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()

def score_candidate(sp_title, sp_artists, sp_duration_ms, yt_result) -> float:
    sp_title_n = normalize(sp_title)
    sp_artists_n = [normalize(a) for a in sp_artists]

    yt_title = yt_result.get("title", "")
    yt_title_n = normalize(yt_title)

    title_score = fuzz.token_set_ratio(sp_title_n, yt_title_n) / 100.0

    yt_artists = yt_result.get("artists") or []
    yt_artist_names = [normalize(a.get("name", "")) for a in yt_artists]
    artist_score = 0.0
    for a in sp_artists_n:
        for b in yt_artist_names:
            artist_score = max(artist_score, fuzz.token_set_ratio(a, b) / 100.0)

    sp_sec = (sp_duration_ms or 0) / 1000.0
    yt_duration = yt_result.get("duration")
    yt_sec = None
    if yt_duration:
        m = re.match(r"(\d+):(\d+)", yt_duration)
        if m:
            yt_sec = int(m.group(1)) * 60 + int(m.group(2))
    dur_score = 0.7
    if yt_sec is not None and sp_sec > 0:
        diff = abs(yt_sec - sp_sec)
        dur_score = max(0.0, 1.0 - min(diff, 10) / 10.0)

    base = 0.55 * title_score + 0.35 * artist_score + 0.10 * dur_score
    if yt_result.get("resultType") == "song":
        base += 0.05
    return min(base, 1.0)

def _diag_from_best(best, best_score, last_query):
    try:
        best_title = best.get("title") if best else None
        best_artists = [a.get("name") for a in (best.get("artists") or [])] if best else None
        return {
            "best_title": best_title,
            "best_artists": best_artists,
            "best_score": round(float(best_score or 0), 3),
            "last_query": last_query,
        }
    except Exception:
        return {"best_title": None, "best_artists": None, "best_score": 0.0, "last_query": last_query}

def yt_best_match(yt: YTMusic, name: str, artists: List[str], duration_ms: Optional[int]):
    """
    Resilient search: swallows JSON parsing errors from ytmusicapi,
    tries multiple queries/filters, returns (best, diag).
    """
    queries = [
        f'"{name}" {artists[0] if artists else ""}',
        f"{name} {artists[0] if artists else ''}",
        f"{name}",
    ]
    best = None
    best_score = 0.0
    last_q = None
    last_err = None

    for q in queries:
        last_q = q
        for flt in ("songs", "videos"):
            try:
                results = yt.search(q, filter=flt) or []
            except Exception as e:
                last_err = str(e)
                time.sleep(0.15)
                continue

            for r in results[:8]:
                sc = score_candidate(name, artists, duration_ms, r)
                if sc > best_score:
                    best = r
                    best_score = sc
            if best_score >= 0.92:
                break
        if best_score >= 0.92:
            break

    diag = _diag_from_best(best, best_score, last_q)
    if last_err and not best:
        diag["last_error"] = last_err

    if best and best_score >= 0.60:
        best["matchScore"] = round(best_score, 3)
        return best, diag

    return None, diag

def spotify_liked_tracks(sp: spotipy.Spotify) -> List[Dict[str, Any]]:
    items = []
    limit = 50
    offset = 0
    while True:
        page = sp.current_user_saved_tracks(limit=limit, offset=offset)
        for it in page["items"]:
            t = it["track"]
            if not t:
                continue
            items.append({
                "id": t["id"],
                "name": t["name"],
                "artists": [a["name"] for a in t.get("artists", [])],
                "duration_ms": t.get("duration_ms"),
                "album": t.get("album", {}).get("name"),
            })
        if page["next"]:
            offset += limit
        else:
            break
    return items

# ---------- Raw YT endpoints (no ytmusicapi JSON parsing) ----------

def _load_resolved_headers() -> Dict[str, str]:
    """Read the last resolved headers we send to YT (written by get_yt)."""
    try:
        with open(".yt_headers_resolved.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Rebuild via get_yt(), which writes the file
        _ = get_yt()
        with open(".yt_headers_resolved.json", "r", encoding="utf-8") as f:
            return json.load(f)

def _fetch_ytcfg_and_key(headers: Dict[str, str]) -> Dict[str, Any]:
    """
    GET music.youtube.com and parse ytcfg JSON to obtain:
      - INNERTUBE_API_KEY (query param 'key')
      - INNERTUBE_CONTEXT (request body 'context')
      - VISITOR_DATA (optional, for X-Goog-Visitor-Id)
    """
    h = dict(headers)
    h.setdefault("Referer", "https://music.youtube.com/")
    h.setdefault("Accept-Language", "en-US,en;q=0.9")
    r = requests.get("https://music.youtube.com", headers=h, timeout=20)
    if r.status_code != 200:
        raise HTTPException(400, f"YT homepage fetch failed: HTTP {r.status_code}")

    m = re.search(r"ytcfg\.set\((\{.*?\})\);", r.text, re.DOTALL)
    if not m:
        raise HTTPException(400, "Could not locate ytcfg JSON on music.youtube.com")
    cfg = json.loads(m.group(1))

    key = cfg.get("INNERTUBE_API_KEY")
    context = cfg.get("INNERTUBE_CONTEXT")
    visitor = cfg.get("VISITOR_DATA")
    if not key or not context:
        raise HTTPException(400, "ytcfg missing API key/context (are headers valid for music.youtube.com?)")
    return {"key": key, "context": context, "visitor": visitor}

def _yt_api_post(endpoint: str, headers: Dict[str, str], key: str, context: Dict[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST to a YT Music internal endpoint and return JSON.
    - Refreshes SAPISIDHASH per request.
    - If we hit a 401/403 or HTML "Sorry" page, re-fetches ytcfg (key/context/visitor) and retries once.
    """
    def _post_once(h, k, ctx):
        url = f"https://music.youtube.com/youtubei/v1/{endpoint}?prettyPrint=false&key={k}"
        payload = {"context": ctx}
        payload.update(body)
        resp = requests.post(url, headers=_freshen_headers_for_post(h), json=payload, timeout=25)
        return resp

    # First attempt
    resp = _post_once(headers, key, context)
    try:
        return resp.json()
    except JSONDecodeError:
        # If non-JSON AND potentially blocked, try a one-time ytcfg refresh + retry
        blocked = resp.status_code in (401, 403) or "<html" in (resp.text[:20].lower() if resp.text else "")
        if not blocked:
            snippet = (resp.text or "")[:300]
            raise HTTPException(400, f"YT {endpoint} returned non-JSON (HTTP {resp.status_code}): {snippet}")

        # Refresh key/context/visitor and retry once
        try:
            refreshed_headers = dict(headers)
            cfg = _fetch_ytcfg_and_key(refreshed_headers)
            new_key, new_ctx, visitor = cfg["key"], cfg["context"], cfg["visitor"]
            if visitor:
                refreshed_headers["X-Goog-Visitor-Id"] = visitor
            resp2 = _post_once(refreshed_headers, new_key, new_ctx)
            return resp2.json()
        except JSONDecodeError:
            snippet = (resp2.text or "")[:300]
            raise HTTPException(403, f"YT {endpoint} returned non-JSON after retry (HTTP {resp2.status_code}): {snippet}")

def _yt_create_playlist_raw(title: str, description: str, privacy: str = "PRIVATE"):
    """
    Create playlist via raw endpoint. Returns (playlist_id, headers, key, context).
    """
    headers = _load_resolved_headers()
    cfg = _fetch_ytcfg_and_key(headers)
    key, context, visitor = cfg["key"], cfg["context"], cfg["visitor"]

    if visitor:
        headers["X-Goog-Visitor-Id"] = visitor

    j = _yt_api_post("playlist/create", headers, key, context, {
        "title": title,
        "description": description,
        "privacyStatus": privacy
    })
    pid = j.get("playlistId")
    if not pid:
        raise HTTPException(400, f"YT playlist/create did not return playlistId: {list(j.keys())}")
    return pid, headers, key, context

def _yt_add_playlist_items_raw(playlist_id: str, video_ids: List[str],
                               headers: Dict[str, str], key: str, context: Dict[str, Any]) -> int:
    """
    Add items via raw endpoints, with pacing + resilient fallbacks.
    1) Try /playlist/add_item (array of videoIds) in small chunks.
    2) Fallback to /browse/edit_playlist (ACTION_ADD_VIDEO) per chunk.
    3) Final fallback: ytmusicapi.add_playlist_items for stubborn cases.
    """
    count = 0
    CH = 25  # smaller chunks to avoid anti-bot wall
    for i in range(0, len(video_ids), CH):
        chunk = [vid for vid in video_ids[i:i+CH] if vid]
        if not chunk:
            continue

        # First try: /playlist/add_item (bulk)
        try:
            _ = _yt_api_post("playlist/add_item", headers, key, context, {
                "playlistId": playlist_id,
                "videoIds": chunk
            })
            count += len(chunk)
            time.sleep(0.25)  # gentle pacing
            continue
        except HTTPException as e1:
            # Second try: /browse/edit_playlist with actions
            try:
                actions = [{"action": "ACTION_ADD_VIDEO", "addedVideoId": vid} for vid in chunk]
                _ = _yt_api_post("browse/edit_playlist", headers, key, context, {
                    "playlistId": playlist_id,
                    "actions": actions
                })
                count += len(chunk)
                time.sleep(0.35)
                continue
            except HTTPException as e2:
                # Final fallback: ytmusicapi (uses its own request plumbing)
                try:
                    yt = get_yt()
                    yt.add_playlist_items(playlist_id, chunk, duplicates=True)
                    count += len(chunk)
                    time.sleep(0.35)
                    continue
                except Exception as e3:
                    # Surface the first hard failure; include status/snippet if present
                    raise HTTPException(400, f"YT add items failed (chunk starting @{i}): "
                                              f"add_item={str(e1)} | edit_playlist={str(e2)} | ytmusicapi={str(e3)}")
    return count

# ---------- Offline (Kaggle dataset) recommender ----------

# Try the richest file first; the helper will also try with/without ".csv"
# Try track-level first; then fall back to artist-level

_DATASET_CANDIDATES = [
    "data_w_genres",   # usually has per-track rows + genres
    "data",            # per-track rows (sometimes without genres)
    "data_by_year",    # aggregated; usually no track names
    "data_by_genre",   # aggregated; usually no track names
    "data_by_artist",  # aggregated per artist (fallback mode)
]

# prefer these numeric features if present (we’ll use what exists)
_NUMERIC_FEATURES_PREF = [
    "danceability","energy","valence","acousticness",
    "instrumentalness","liveness","speechiness",
    "tempo","loudness","key","mode","time_signature",
    "popularity","count"  # added so artist/genre/year tables still have signal
]

_OFFLINE_CACHE = {
    "df": None,
    "X": None,             # standardized numeric features
    "features": None,      # feature column names
    "track_col": None,     # e.g., 'track_name' or 'name'
    "artist_col": None,    # e.g., 'artists' or 'artist_name'
    "genre_col": None,     # e.g., 'genres' or 'genre'
}

_NUMERIC_FEATURES_PREF = [
    "danceability","energy","valence","acousticness",
    "instrumentalness","liveness","speechiness",
    "tempo","loudness","key","mode","time_signature",
]

def _path_exists(p):
    return os.path.isfile(p) and os.path.getsize(p) > 0

def _first_existing_path(base):
    p1 = os.path.join(DATA_DIR, base)
    p2 = os.path.join(DATA_DIR, f"{base}.csv")
    if _path_exists(p1):
        return p1
    if _path_exists(p2):
        return p2
    return None

def _pick_col(df_cols, options):
    s = set(c.lower() for c in df_cols)
    for name in options:
        if name in s:
            for c in df_cols:
                if c.lower() == name:
                    return c
    return None

def _ensure_norm_cols(df, track_col, artist_col):
    if "track_norm" not in df.columns:
        df["track_norm"] = df[track_col].astype(str).map(normalize)
    if "artist_norm" not in df.columns:
        def _first_artist(s):
            s = str(s)
            for delim in ["','", '", "', ";", "|", ","]:
                if delim in s:
                    s = s.split(delim)[0]
                    break
            s = re.sub(r"^\[?['\"]?", "", s)
            s = re.sub(r"['\"]?\]?$", "", s)
            return s
        df["artist_norm"] = df[artist_col].map(_first_artist).map(normalize)

def _load_offline_dataset():
    """
    Load one of the Kaggle CSVs from DATA_DIR.
    - Prefer a track-level file (with a track title column).
    - If no track-level file exists, fall back to artist-level file and set mode='artist'.
    Caches: df, X (standardized feature matrix), features, track_col (or None), artist_col, genre_col, mode
    """
    if _OFFLINE_CACHE.get("df") is not None:
        return _OFFLINE_CACHE

    tried = []
    for cand in _DATASET_CANDIDATES:
        path = _first_existing_path(cand)
        if not path:
            continue

        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        tried.append((os.path.basename(path), list(df.columns)))

        track_col = _pick_col(df.columns, ["track_name","name","track","title","song"])
        artist_col = _pick_col(df.columns, ["artists","artist_name","artist"])
        genre_col  = _pick_col(df.columns, ["genres","genre","tags"])

        # need at least artist column
        if not artist_col:
            continue

        # pick usable numeric features that exist in this file
        feats_lower = [c.lower() for c in df.columns]
        feats = [f for f in _NUMERIC_FEATURES_PREF if f in feats_lower]
        feats = [next(cc for cc in df.columns if cc.lower() == f) for f in feats]
        if not feats:
            continue

        # numeric matrix
        num = df[feats].copy()
        for c in num.columns:
            num[c] = pd.to_numeric(num[c], errors="coerce").fillna(0.0)

        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(num.values.astype("float32"))

        # Track-level mode
        if track_col:
            _ensure_norm_cols(df, track_col, artist_col)
            _OFFLINE_CACHE.update({
                "df": df, "X": X, "features": feats,
                "track_col": track_col, "artist_col": artist_col, "genre_col": genre_col,
                "mode": "track",
            })
            return _OFFLINE_CACHE

        # Artist-level fallback (no track titles in this file)
        df["artist_norm"] = df[artist_col].astype(str).map(normalize)
        _OFFLINE_CACHE.update({
            "df": df, "X": X, "features": feats,
            "track_col": None, "artist_col": artist_col, "genre_col": genre_col,
            "mode": "artist",
        })
        return _OFFLINE_CACHE

    raise HTTPException(
        500,
        "No usable Kaggle CSV found in the 'data' folder.\n"
        f"Tried (and saw columns): {[(n, cols[:8] + ['...']) for n, cols in tried]}.\n"
        "Place at least one of these files: "
        f"{_DATASET_CANDIDATES} (.csv) with numeric audio features."
    )

def _match_liked_to_dataset(liked_items, df):
    """
    liked_items: [{'name': str, 'artists': [str,...]}, ...]
    returns: set of row indices in df that match liked tracks by title+artist
    """
    by_title = {}
    track_norm_vals = df["track_norm"].values
    for idx, t in enumerate(track_norm_vals):
        by_title.setdefault(t, []).append(idx)

    matched = set()
    for t in liked_items:
        tname = normalize(t.get("name") or "")
        tart = normalize((t.get("artists") or [""])[0] or "")
        if not tname:
            continue

        cand = by_title.get(tname, [])
        best_idx, best_score = None, 0.0

        for i in cand:
            a = str(df.at[i, "artist_norm"] or "")
            s = fuzz.token_set_ratio(tart, a) / 100.0 if tart else 0.0
            if s > best_score:
                best_idx, best_score = i, s

        if best_idx is None:
            first = tname[:1]
            pool = [i for i in range(len(df)) if str(df.at[i, "track_norm"]).startswith(first)]
            for i in pool:
                title_s = fuzz.token_set_ratio(tname, str(df.at[i, "track_norm"])) / 100.0
                if title_s >= 0.90:
                    a = str(df.at[i, "artist_norm"] or "")
                    s = fuzz.token_set_ratio(tart, a) / 100.0 if tart else 0.0
                    combo = 0.7*title_s + 0.3*s
                    if combo > best_score:
                        best_idx, best_score = i, combo

        if best_idx is not None and best_score >= 0.70:
            matched.add(best_idx)

    return matched

def _match_artists_to_dataset(liked_items, df):
    """
    Fallback match when the dataset is artist-aggregated (no track titles):
    returns set of row indices in df for artists that appear in your likes.
    """
    # Build quick index: normalized artist name -> list of row indices
    idx = {}
    for i in range(len(df)):
        a = str(df.at[i, "artist_norm"] or "")
        if a:
            idx.setdefault(a, []).append(i)

    matched = set()
    for t in liked_items:
        for a in (t.get("artists") or []):
            key = normalize(a)
            for i in idx.get(key, []):
                matched.add(i)
    return matched

def _cosine_sim_matrix_to_vec(M, v):
    mv = M @ v
    M_norm = np.linalg.norm(M, axis=1) + 1e-12
    v_norm = np.linalg.norm(v) + 1e-12
    return (mv / (M_norm * v_norm))

# ---------- Routes ----------

@app.get("/api/health")
def health():
    here = os.getcwd()
    oauth_files = [f for f in os.listdir(here) if f.lower().endswith(".json") and "oauth" in f.lower()]
    return {"ok": True, "frontend_base": FRONTEND_BASE, "cwd": here, "oauth_files": oauth_files}

@app.get("/api/spotify/login")
def spotify_login():
    _check_spotify_env()
    oauth = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=" ".join(SCOPES),
        cache_path=".spotify_cache",
        show_dialog=True,
    )
    auth_url = oauth.get_authorize_url()
    return {"auth_url": auth_url}

@app.get("/api/spotify/callback")
def spotify_callback(code: Optional[str] = None, error: Optional[str] = None):
    if error:
        raise HTTPException(400, f"Spotify error: {error}")
    _check_spotify_env()
    oauth = SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=" ".join(SCOPES),
        cache_path=".spotify_cache",
        show_dialog=False,
    )
    token_info = oauth.get_access_token(code, check_cache=False)
    STATE["spotify_token"] = token_info
    return RedirectResponse(url=f"{FRONTEND_BASE}?spotify=connected")

@app.post("/api/ytmusic/connect")
def ytmusic_connect(payload: YTHeadersPayload):
    lowered = {(k.lower() if isinstance(k, str) else k): v for k, v in payload.headers_json.items()}
    if "cookie" not in lowered or "user-agent" not in lowered:
        raise HTTPException(400, "Headers must include 'cookie' and 'user-agent' from music.youtube.com")
    lowered.setdefault("x-origin", "https://music.youtube.com")
    lowered.setdefault("x-goog-authuser", "0")

    STATE["yt_headers"] = lowered
    with open(".yt_headers.json", "w", encoding="utf-8") as f:
        json.dump(lowered, f, ensure_ascii=False, indent=2)

    try:
        yt = get_yt()
        _ = yt.get_library_playlists(limit=1)
    except Exception as e:
        raise HTTPException(400, f"Saved headers but YouTube rejected them: {e}")

    return {"status": "connected"}

@app.get("/api/ytmusic/status")
def ytmusic_status():
    try:
        yt = get_yt()
        try:
            _ = yt.get_library_playlists(limit=1)
        except TypeError:
            return {"connected": True}
        return {"connected": True}
    except HTTPException as he:
        return {"connected": False, "reason": str(he.detail) if hasattr(he, "detail") else str(he)}
    except Exception as e:
        return {"connected": False, "reason": str(e)}

@app.get("/api/spotify/status")
def spotify_status():
    try:
        sp = get_spotify()
        me = sp.me()
        return {"connected": True, "user": me.get("display_name") or me.get("id")}
    except Exception:
        return {"connected": False}

@app.post("/api/export", response_model=ExportResult)
def export_library(req: ExportRequest):
    try:
        sp = get_spotify()
        yt = get_yt()  # validate headers & write .yt_headers_resolved.json

        liked = spotify_liked_tracks(sp)
        total = len(liked)
        if total == 0:
            raise HTTPException(400, "No Liked Songs on Spotify")

        playlist_name = req.playlist_name or f"Spotify Liked Songs ({time.strftime('%Y-%m-%d')})"

        try:
            playlist_id, yth, ykey, yctx = _yt_create_playlist_raw(
                playlist_name,
                description="Imported from Spotify Liked Songs",
                privacy="PRIVATE"
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"YT create_playlist failed: {e}")

        matched_items: List[Dict[str, Any]] = []
        unmatched: List[Dict[str, Any]] = []

        for t in liked:
            try:
                m, diag = yt_best_match(yt, t["name"], t["artists"], t["duration_ms"])
            except Exception as e:
                diag = {"best_title": None, "best_artists": None, "best_score": 0.0,
                        "last_query": None, "last_error": str(e)}
                m = None

            if m:
                matched_items.append(m)
            else:
                unmatched.append({
                    "name": t["name"],
                    "artists": t["artists"],
                    "album": t.get("album"),
                    "duration_ms": t.get("duration_ms"),
                    "diagnostic": diag,
                })

        added = 0
        video_ids = [c.get("videoId") for c in matched_items if c.get("videoId")]
        try:
            added = _yt_add_playlist_items_raw(playlist_id, video_ids, yth, ykey, yctx)
        except HTTPException as e:
            raise HTTPException(400, f"YT add items failed: {e.detail}")
        except Exception as e:
            raise HTTPException(400, f"YT add items failed: {e}")

        liked_count = 0
        if req.also_like_on_ytm:
            for vid in video_ids:
                try:
                    yt.rate_song(vid, "LIKE")
                    liked_count += 1
                    time.sleep(0.05)
                except Exception:
                    pass

        return ExportResult(
            created_playlist_id=playlist_id,
            total_spotify_tracks=total,
            matched=len(matched_items),
            added=added,
            liked=liked_count,
            unmatched_samples=unmatched[:10],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YT export failed: {e}")

@app.post("/api/recommendations_offline")
def recommendations_offline(req: RecsRequest):
    """
    If we loaded a track-level CSV (with track titles): match on (title+artist) and recommend similar tracks.
    If only an artist-level CSV exists: match liked artists, recommend similar artists, and
    fill a representative top track for each recommended artist via Spotify (metadata only).
    """
    sp = get_spotify()

    # try to get a market for nicer top-tracks
    try:
        me = sp.me() or {}
        market = me.get("country") or "US"
    except Exception:
        market = "US"

    liked = spotify_liked_tracks(sp)
    if not liked:
        raise HTTPException(400, "No Liked Songs on Spotify to match.")

    cache = _load_offline_dataset()
    df, X, mode = cache["df"], cache["X"], cache["mode"]
    N = min(100, max(1, int(req.count or 30)))

    if mode == "track":
        matched_idx = _match_liked_to_dataset(liked, df)
        if not matched_idx:
            raise HTTPException(
                502,
                "Could not match any Liked Songs to the offline dataset's tracks. "
                "Tip: ensure 'data_w_genres.csv' or 'data.csv' in /data includes track titles."
            )

        idx_list = sorted(list(matched_idx))
        centroid = X[idx_list].mean(axis=0)
        sims = _cosine_sim_matrix_to_vec(X, centroid)
        sims[idx_list] = -1.0  # exclude already-liked matches

        top_idx = np.argpartition(-sims, N)[:N]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        out = []
        track_col = cache["track_col"]
        artist_col = cache["artist_col"]
        genre_col  = cache["genre_col"]
        album_col = next((c for c in df.columns if c.lower() == "album"), None)

        for i in top_idx:
            name = str(df.at[i, track_col])
            artists_raw = str(df.at[i, artist_col])
            artists = []
            if artists_raw:
                tmp = re.sub(r"^\[|\]$", "", artists_raw)
                tmp = tmp.replace("'", "").replace('"', "")
                artists = [x.strip() for x in re.split(r"[|;,]", tmp) if x.strip()] or [artists_raw]

            rec = {
                "name": name,
                "artists": artists,
                "album": (str(df.at[i, album_col]) if album_col else ""),
                "score": float(round(sims[i], 4)),
            }
            if genre_col:
                rec["genres"] = str(df.at[i, genre_col])
            out.append(rec)

        return {"count": len(out), "suggestions": out}

    # -------- artist fallback mode ----------
    # Match liked artists → rows, then recommend similar artists and attach a top track title via Spotify
    if "artist_norm" not in df.columns:
        df["artist_norm"] = df[cache["artist_col"]].astype(str).map(normalize)

    matched_idx = _match_artists_to_dataset(liked, df)
    if not matched_idx:
        raise HTTPException(
            502,
            "Could not match any liked artists to the offline dataset (artist mode). "
            "Open a few liked songs in Spotify and make sure the artist names exist in the Kaggle file."
        )

    idx_list = sorted(list(matched_idx))
    centroid = X[idx_list].mean(axis=0)
    sims = _cosine_sim_matrix_to_vec(X, centroid)
    sims[idx_list] = -1.0  # exclude already-liked artists

    top_idx = np.argpartition(-sims, N)[:N]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    out = []
    artist_col = cache["artist_col"]
    genre_col  = cache["genre_col"]

    for i in top_idx:
        artist_name = str(df.at[i, artist_col])

        # Best-effort: get a representative top track for this artist (title/album only)
        title, album = f"Discover {artist_name}", ""
        try:
            # find artist id
            sr = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
            items = (sr.get("artists") or {}).get("items") or []
            if items and items[0].get("id"):
                aid = items[0]["id"]
                tops = sp.artist_top_tracks(aid, country=market) or {}
                tlist = tops.get("tracks") or []
                if tlist:
                    title = tlist[0].get("name") or title
                    album = (tlist[0].get("album") or {}).get("name") or ""
        except Exception:
            pass  # keep the fallback title

        rec = {
            "name": title,
            "artists": [artist_name],
            "album": album,
            "score": float(round(sims[i], 4)),
        }
        if genre_col:
            rec["genres"] = str(df.at[i, genre_col])
        out.append(rec)

    return {"count": len(out), "suggestions": out}

# Simple debug endpoints

@app.get("/api/ytmusic/resolved")
def ytmusic_resolved():
    try:
        data = _load_resolved_headers()
    except Exception as e:
        return {"have_resolved": False, "reason": str(e)}
    cookie = data.get("Cookie") or ""
    return {
        "have_resolved": True,
        "keys": list(data.keys()),
        "user_agent_len": len(data.get("User-Agent", "") or ""),
        "auth_present": bool(data.get("Authorization")),
        "cookie_has_sapisid": any(tok in cookie for tok in ["SAPISID","__Secure-3PAPISID","__Secure-3PSID"]),
        "cookie_sample": (cookie[:80] + "...(truncated)") if cookie else "",
    }

@app.get("/api/ytmusic/smoke_raw")
def ytmusic_smoke_raw():
    """
    Raw write smoke: fetch key/context from ytcfg, then POST playlist/create.
    """
    try:
        headers = _load_resolved_headers()
        meta = _fetch_ytcfg_and_key(headers)
        key, context, visitor = meta["key"], meta["context"], meta["visitor"]

        post_headers = dict(headers)
        post_headers["Content-Type"] = "application/json"
        post_headers["Accept"] = "application/json"
        if visitor:
            post_headers["X-Goog-Visitor-Id"] = visitor

        name = f"SMOKE TEST {int(time.time())}"
        body = {
            "context": context,
            "title": name,
            "description": "temporary smoke test via API",
            "privacyStatus": "PRIVATE"
        }

        url = f"https://music.youtube.com/youtubei/v1/playlist/create?prettyPrint=false&key={key}"
        resp = requests.post(url, headers=post_headers, json=body, timeout=25)
        text = (resp.text or "")[:300]
        out = {"status": resp.status_code, "is_json": False, "text_snippet": text}

        try:
            j = resp.json()
            out["is_json"] = True
            out["json_keys"] = list(j.keys())
            if "playlistId" in j:
                out["playlistId"] = j["playlistId"]
        except JSONDecodeError:
            pass
        return out
    except Exception as e:
        return {"status": None, "is_json": False, "error": str(e)}

# Log exceptions to console so you can see crashes during dev
@app.middleware("http")
async def _log_requests(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        print(f"[SERVER ERROR] {request.method} {request.url}: {e}", flush=True)
        raise
