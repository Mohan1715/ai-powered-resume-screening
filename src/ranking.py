import pandas as pd

def rank_resumes(resume_ids, scores):
    df = pd.DataFrame({
        "Resume_ID": resume_ids,
        "Score": scores.flatten()
    })

    df = df.sort_values(by="Score", ascending=False)
    df["Score"] = df["Score"] * 100  # convert to percentage
    df.reset_index(drop=True, inplace=True)

    return df