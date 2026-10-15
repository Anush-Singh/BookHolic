# 📚 BookHolic — Novel Recommendation System

**Discover your next favourite book, one page at a time.**

BookHolic is a book recommendation web application designed to help readers discover novels based on their preferred genre, reading experience, and most recently read book. It combines machine learning-based recommendations with book-series discovery to make finding your next read easier.

## ✨ Features

- **Personalized Recommendations:** Get book suggestions based on your selected genre and reading experience.
- **Genre-Based Discovery:** Explore Romance, Fantasy, Mystery, Thriller, Fiction, and Self-Help.
- **Same-Series Recommendations:** Discover subsequent books from a series when supported by available book metadata.
- **Machine Learning:** Uses TF-IDF vectorization and cosine similarity to identify books with similar textual features.
- **Open Library Integration:** Retrieves book information and cover images through the Open Library Search API.
- **Interactive Website:** Browse recommendations through a vintage-library-inspired interface.
- **REST API:** A FastAPI backend processes recommendation requests.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, FastAPI |
| Machine Learning | Scikit-learn, TF-IDF, Cosine Similarity |
| Data Processing | Pandas, NumPy |
| Book Data API | Open Library |
| Model Storage | Pickle (`.pkl`) |
| Version Control | Git and GitHub |

## 🏗️ Project Structure

```text
BookHolic/
├── index.html             # Website interface
├── main.py                # FastAPI recommendation backend
├── requirements.txt       # Python dependencies
├── bookholic_model.pkl    # Trained recommendation model
├── README.md              # Project documentation
└── .gitignore             # Git exclusions
```

## ⚙️ Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- Git
- A modern web browser

### 1. Clone the repository

```bash
git clone https://github.com/Anush-Singh/BookHolic.git
cd BookHolic
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the backend

Ensure `bookholic_model.pkl` is present in the project root.

```bash
uvicorn main:app --reload
```

The local API runs at:

- API base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/health`

### 5. Open the frontend

Open `index.html` using the VS Code Live Server extension or another local static web server.

For local testing, the frontend should point to:

```javascript
const API_URL = "http://127.0.0.1:8000/recommend";
```

For public deployment, replace this URL with the deployed backend's API endpoint.

## 🔌 Recommendation API

### Endpoint

```http
POST /recommend
```

### Example request

```json
{
  "genre": "Romance",
  "experience": "beginner",
  "latest_book": "Twisted Love",
  "n": 10
}
```

### Response

The API returns a JSON object containing the matched book, the number of recommendations, and the recommendation list. Individual recommendation fields depend on the backend response.

Use the interactive API documentation at `/docs` to test the endpoint and inspect its actual response schema.

## 🧠 How It Works

1. **User Input:** The reader selects a preferred genre, reading experience, and latest-read book.
2. **Book Matching:** The backend attempts to match the submitted title against available book information.
3. **Series Discovery:** When suitable series metadata is available, later books in the same series can be prioritized.
4. **Text Similarity:** TF-IDF features and cosine similarity help rank relevant titles from the local recommendation model.
5. **Additional Book Data:** Open Library can supply book metadata and cover information when needed.
6. **Recommendations:** The backend returns book suggestions for display on the website.

*Note: Recommendation quality depends on the dataset, metadata coverage, and model features. Reading experience is a heuristic input, not a validated measurement of reading difficulty.*

## 🌐 Deployment

The planned deployment architecture is:

- **Frontend:** GitHub Pages
- **Backend API:** Render
- **Source Code:** GitHub

The frontend and backend must be connected using the deployed API URL. Deployment is complete only after the live website successfully receives recommendations from the hosted backend.

## 🔮 Future Improvements

- Improve genre classification and book metadata quality.
- Add stronger reader-preference personalization.
- Improve book-series ordering and metadata coverage.
- Add book ratings, wishlists, and reading history.
- Improve error handling and API response caching.
- Enhance the recommendation model using richer datasets.

## 👩‍💻 Author

**Anush Singh**

- GitHub: [Anush-Singh](https://github.com/Anush-Singh)
- Project Repository: [BookHolic](https://github.com/Anush-Singh/BookHolic)
---
*BookHolic — Find the story that feels like it was written for you.*