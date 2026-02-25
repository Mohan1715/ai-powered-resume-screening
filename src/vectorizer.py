from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_matrix(resume_texts, job_description):
    documents = resume_texts + [job_description]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    return tfidf_matrix