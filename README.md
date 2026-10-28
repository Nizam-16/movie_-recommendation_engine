# Movie Recommendation Engine Mini Project

This is a Python-based **Content-Based Movie Recommendation Engine** that uses **TF-IDF Vectorization** and **Cosine Similarity** to suggest movies based on genre and plot descriptions.

## Setup Instructions

1. **Prerequisites:** Make sure you have Python installed (Python 3.7+ recommended).

2. **Install Required Packages:**
   ```bash
   pip install pandas scikit-learn
   ```

3. **Run the Project:**
   ```bash
   python recommender.py
   ```

## Project Features
- **Built-in Dataset:** Pre-loaded with popular movies, genres, and descriptions.
- **Interactive CLI:** Search for movies, view recommendations with match percentages, or list all available movies.
- **Content-Based Filtering:** Uses Scikit-learn's `TfidfVectorizer` and `cosine_similarity` algorithm.
