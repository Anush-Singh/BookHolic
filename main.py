import pickle
from pathlib import Path
from difflib import get_close_matches
from functools import lru_cache
import re

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. FastAPI application
# -----------------------------
app = FastAPI(title="BookHolic API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this before production deployment
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
OPEN_LIBRARY_TIMEOUT = 8
USER_AGENT = "BookHolic/1.0 (book recommendation demo)"

# -----------------------------
# 2. Load trained local model
# -----------------------------
MODEL_PATH = Path(__file__).parent / "bookholic_model.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    books = model["popular_books"].reset_index(drop=True)
    tfidf_matrix = model["tfidf_matrix"]
except Exception as exc:
    raise RuntimeError(
        f"Could not load model: {exc}. Check bookholic_model.pkl and the "
        "scikit-learn version used to create it."
    ) from exc


# -----------------------------
# 3. Request schema
# -----------------------------
class RecommendationRequest(BaseModel):
    latest_book: str = Field(min_length=1, max_length=200)
    genre: str = Field(min_length=1, max_length=80)
    experience: str = "Beginner"
    n: int = Field(default=10, ge=1, le=10)


# -----------------------------
# 4. Helpers
# -----------------------------
def normalize_genre(genre):
    value = genre.strip().lower()
    aliases = {
        "thriller": "mystery/thriller",
        "mystery": "mystery/thriller",
        "fiction": "general fiction",
        "science fiction": "general fiction",
        "horror": "general fiction",
        "biography": "general fiction",
        "history": "general fiction",
    }
    return aliases.get(value, value)


def normalize_experience(experience):
    value = experience.strip().lower()
    if value in {"beginner", "new reader", "0"}:
        return "beginner"
    if value in {"intermediate", "2-5 years", "2 to 5 years", "5"}:
        return "intermediate"
    return "experienced"


def experience_score(rating_count, experience):
    count = int(rating_count or 0)
    if experience == "beginner":
        return 1.0 if count < 50 else 0.7
    if experience == "intermediate":
        return 1.0 if 10 <= count <= 200 else 0.7
    return 1.0 if count > 50 else 0.7


def cover_url(cover_id):
    return f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg" if cover_id else ""


def first_value(value, default=""):
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value) if value is not None else default


@lru_cache(maxsize=256)
def open_library_search(query, limit=25):
    """Cached Open Library search. Empty results are cached too."""
    params = {
        "q": query,
        "limit": limit,
        "fields": "key,title,author_name,cover_i,first_publish_year,series_name,series_position,subject",
    }
    try:
        response = requests.get(
            OPEN_LIBRARY_SEARCH,
            params=params,
            timeout=OPEN_LIBRARY_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("docs", [])
    except (requests.RequestException, ValueError):
        return []


def search_open_library_title(title):
    """Return the best Open Library title match."""
    docs = open_library_search(f'title:"{title}"', 10)
    if not docs:
        docs = open_library_search(title, 10)
    if not docs:
        return None

    target = title.casefold().strip()
    exact = [doc for doc in docs if str(doc.get("title", "")).casefold().strip() == target]
    return exact[0] if exact else docs[0]


def parse_position(value):
    """Extract a sortable series number; e.g. '2' -> 2, '1-4' -> 1."""
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else None


def doc_to_recommendation(doc, genre="", source="Open Library"):
    authors = doc.get("author_name") or []
    cover_id = doc.get("cover_i")
    return {
        "Book-Title": str(doc.get("title") or "Unknown title"),
        "Book-Author": ", ".join(map(str, authors)) if authors else "Unknown author",
        "Genre": genre or first_value(doc.get("subject"), "General Fiction"),
        "Image-URL-M": cover_url(cover_id),
        "avg_rating": None,
        "source": source,
        "first_publish_year": doc.get("first_publish_year"),
    }


def get_series_recommendations(start_doc, requested_title, limit):
    """
    Find later books in the same series when Open Library exposes series metadata.
    API series metadata can be incomplete, so this returns an empty list if it
    cannot confidently identify the series.
    """
    series_name = first_value(start_doc.get("series_name"))
    if not series_name:
        return []

    current_position = parse_position(first_value(start_doc.get("series_position")))
    author = first_value(start_doc.get("author_name"))
    if author:
        docs = open_library_search(
            f'author:"{author}"',100)
    else:
        docs = open_library_search(series_name, 100)

    candidates = []
    seen = {requested_title.casefold().strip()}
    for doc in docs:
        title = str(doc.get("title") or "").strip()
        if not title or title.casefold() in seen:
            continue

        doc_series = [str(item).casefold() for item in (doc.get("series_name") or [])]
        if series_name.casefold() not in doc_series:
            continue

        # Skip omnibus/collection titles that bundle several books together.
        if "/" in title:
            continue

        position_text = first_value(doc.get("series_position"))
        position = parse_position(position_text)
        if current_position is not None and position is not None and position <= current_position:
            continue

        seen.add(title.casefold())
        candidates.append((position if position is not None else 9999, doc))

    print("Detected series:", series_name)
    print("Series candidates:", [
        (doc.get("title"), doc.get("series_position"))
        for i, doc in candidates])
    return [
        doc_to_recommendation(doc, genre="Same Series")
        for _, doc in candidates[:limit]
    ]


def get_local_recommendations(matched_title, genre, experience, limit, exclude_titles=None):
    titles = books["Book-Title"].fillna("").astype(str).tolist()
    matches = get_close_matches(matched_title, titles, n=1, cutoff=0.6)
    if not matches:
        return []

    local_title = matches[0]
    rows = books.index[books["Book-Title"].astype(str) == local_title].tolist()
    if not rows:
        return []

    similarities = cosine_similarity(tfidf_matrix[rows[0]], tfidf_matrix).flatten()
    results = books.copy()
    results["similarity"] = similarities
    excluded = {local_title.casefold()}
    excluded.update(t.casefold() for t in (exclude_titles or []))
    results = results[~results["Book-Title"].fillna("").astype(str).str.casefold().isin(excluded)]

    selected_genre = normalize_genre(genre)
    results = results[results["Genre"].fillna("").astype(str).str.lower() == selected_genre]
    if results.empty:
        return []

    normalized_experience = normalize_experience(experience)
    results["experience_score"] = results["total_ratings"].apply(
        lambda count: experience_score(count, normalized_experience)
    )
    results["final_score"] = (
        0.6 * results["similarity"]
        + 0.3 * (pd.to_numeric(results["average_rating"], errors="coerce").fillna(0) / 10)
        + 0.1 * results["experience_score"]
    )
    results = results.sort_values(
        ["final_score", "average_rating", "total_ratings"],
        ascending=False,
    ).head(limit)

    recommendations = []
    for _, row in results.iterrows():
        image_url = row.get("Image-URL-M", "")
        if pd.isna(image_url):
            image_url = ""
        recommendations.append({
            "Book-Title": str(row["Book-Title"]),
            "Book-Author": str(row.get("Book-Author", "Unknown author")),
            "Genre": str(row.get("Genre", genre)),
            "Image-URL-M": str(image_url),
            "avg_rating": round(float(row["average_rating"]), 1)
            if pd.notna(row.get("average_rating")) else None,
            "source": "BookHolic ML model",
        })
    return recommendations


# -----------------------------
# 5. Home endpoint
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "BookHolic API is running!",
        "local_model_books": len(books),
        "external_source": "Open Library",
        "endpoints": ["/", "/health", "/recommend"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "open_library": "configured"}


# -----------------------------
# 6. Recommendation endpoint
# -----------------------------
@app.post("/recommend")
def recommend(request: RecommendationRequest):
    entered_title = request.latest_book.strip()
    api_doc = search_open_library_title(entered_title)

    # Series recommendations can work even when the title is absent from CSV.
    series_recommendations = get_series_recommendations(
        api_doc, entered_title, request.n
    ) if api_doc else []

    # Try the local TF-IDF model next. It remains useful for personalized ranking.
    local_recommendations = get_local_recommendations(
        entered_title,
        request.genre,
        request.experience,
        request.n,
        exclude_titles=[item["Book-Title"] for item in series_recommendations],
    )

    # If the local model doesn't know this book, use Open Library to find
    # related title/author/subject matches as a fallback.
    api_fallback = []
    if api_doc and len(series_recommendations) < request.n:
        author = first_value(api_doc.get("author_name"))
        query_parts = [f'author:"{author}"'] if author else []
        subjects = api_doc.get("subject") or []
        query_parts.extend(f'subject:"{subject}"' for subject in subjects[:2])
        if query_parts:
            fallback_docs = open_library_search(" OR ".join(query_parts), 30)
            seen = {
                entered_title.casefold(),
                *(item["Book-Title"].casefold() for item in series_recommendations),
                *(item["Book-Title"].casefold() for item in local_recommendations),
            }
            for doc in fallback_docs:
                title = str(doc.get("title") or "").strip()
                if not title or title.casefold() in seen or "/" in title:
                    continue
                seen.add(title.casefold())
                api_fallback.append(doc_to_recommendation(doc, genre=request.genre))
                if len(api_fallback) >= request.n:
                    break

    # Priority: remaining books in the same series, then local ML, then API fallback.
    combined = series_recommendations + local_recommendations + api_fallback
    unique = []
    seen_titles = {entered_title.casefold()}
    for item in combined:
        key = item["Book-Title"].casefold()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        unique.append(item)
        if len(unique) >= request.n:
            break

    if not unique:
        if not api_doc:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Book not found in Open Library or the local model. "
                    "Check the title spelling and try again."
                ),
            )
        raise HTTPException(
            status_code=404,
            detail=(
                "No recommendations were found. Try another title or genre. "
                "Open Library may not have enough series metadata for this book."
            ),
        )

    return {
        "matched_book": str(api_doc.get("title")) if api_doc else entered_title,
        "series": first_value(api_doc.get("series_name")) if api_doc else "",
        "genre": request.genre,
        "experience": normalize_experience(request.experience),
        "count": len(unique),
        "recommendations": unique,
    }