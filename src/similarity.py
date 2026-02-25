from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(tfidf_matrix, num_resumes):
    jd_vector = tfidf_matrix[-1]
    resume_vectors = tfidf_matrix[:num_resumes]
    
    similarity_scores = cosine_similarity(resume_vectors, jd_vector)
    return similarity_scores