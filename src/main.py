import pandas as pd
from preprocess import clean_text
from vectorizer import create_tfidf_matrix
from similarity import calculate_similarity
from ranking import rank_resumes

# Load dataset
df = pd.read_csv("../data/resumes.csv")

# Load job description
with open("../data/job_description.txt", "r") as file:
    job_description = file.read()

# Clean resumes
df["Cleaned_Resume"] = df["Resume_str"].apply(clean_text)
cleaned_resumes = df["Cleaned_Resume"].tolist()

# Clean job description
cleaned_jd = clean_text(job_description)

# TF-IDF
tfidf_matrix = create_tfidf_matrix(cleaned_resumes, cleaned_jd)

# Similarity
scores = calculate_similarity(tfidf_matrix, len(cleaned_resumes))

# Ranking
ranked_df = rank_resumes(df["ID"], scores)

print("\nTop 10 Ranked Candidates:\n")
print(ranked_df.head(10))