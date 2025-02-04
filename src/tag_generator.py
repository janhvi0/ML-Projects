import spacy
from keybert import KeyBERT

nlp = spacy.load('en_core_web_md')
keybert_model = KeyBERT()

def generate_tags(question):
    """
    Generate relevant tags for a question using KeyBERT and spaCy.
    Returns a list of tags.
    """
    # Extract keywords using KeyBERT
    keywords = keybert_model.extract_keywords(
        question, 
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=3
    )
    
    # Process with spaCy for additional entity recognition
    doc = nlp(question)
    
    # Extract named entities and technical terms
    entities = [ent.text.lower() for ent in doc.ents]
    
    # Combine and clean tags
    all_tags = [kw[0] for kw in keywords] + entities
    
    # Remove duplicates and clean tags
    clean_tags = list(set([
        tag.strip() for tag in all_tags 
        if len(tag.strip()) > 2  # Remove very short tags
    ]))
    
    return clean_tags[:5]  # Limit to top 5 tags

def assign_tags(df):
    """Adds a 'Tags' column to the dataframe."""
    print("Generating tags...")
    df["tags"] = df["Questions"].apply(generate_tags)
    return df
