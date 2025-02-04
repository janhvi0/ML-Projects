import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words='english')

def find_duplicates(questions):
    """
    Identify duplicate questions using TF-IDF and cosine similarity.
    Returns a boolean array indicating duplicate status.
    """
    tfidf_matrix = tfidf.fit_transform(questions)
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Ignore self-similarity by masking the diagonal
    np.fill_diagonal(cosine_sim, 0)
    
    # Mark duplicates if similarity is above 0.85
    is_duplicate = (cosine_sim > 0.85).any(axis=1)

    return is_duplicate

def detect_duplicates(df):
    """Adds a 'Duplicate' column to the dataframe."""
    print("Finding duplicates...")
    df["duplicate"] = find_duplicates(df["Questions"].tolist())
    return df
