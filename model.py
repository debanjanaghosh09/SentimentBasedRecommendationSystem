import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

#Loading clean dataset
reviews_clean = pickle.load(open("reviews_clean.pkl", "rb"))

#Loading User Based Recommendation
user_final_rating = pickle.load(open("user_final_rating.pkl", "rb"))

#Loading the XGBoost Model
loaded_vec = CountVectorizer(vocabulary=pickle.load(open("count_vector.pkl", "rb")))
loaded_tfidf = pickle.load(open("tfidf.pkl", "rb"))
loaded_model = pickle.load(open("XGB_model.pkl", "rb"))

def get_recommendations(user):
    ser_user = user_final_rating.loc[user]
    df_new = pd.DataFrame({'Id': ser_user.index, 'Score': ser_user.values})
    recom = list(df_new.sort_values('Score', ascending=False)[0:20].Id)
    new_df = reviews_clean[reviews_clean.id.isin(recom)]
    
    X_new_counts = loaded_vec.transform(new_df["reviews_text_clean"])
    X_new_tfidf = loaded_tfidf.transform(X_new_counts)
    predicted = loaded_model.predict(X_new_tfidf)
    new_df["predicted_sentiment"] = predicted
    new_df = new_df[['id', 'predicted_sentiment']]
    final_df = new_df.groupby('id', as_index=False).count()
    final_df["negative_review_count"] = final_df.id.apply(lambda x: new_df[(new_df.id==x) & (new_df.predicted_sentiment==0)]["predicted_sentiment"].count())
    final_df["positive_review_count"] = final_df.id.apply(lambda x: new_df[(new_df.id==x) & (new_df.predicted_sentiment==1)]["predicted_sentiment"].count())
    final_df["total_review_count"] = final_df['predicted_sentiment']
    final_df['positive_sentiment_percent'] = np.round(final_df["positive_review_count"]/final_df["total_review_count"]*100,2)
    top5products = final_df.sort_values('positive_sentiment_percent', ascending=False)[0:5]
    return pd.merge(top5products, reviews_clean, left_on='id', right_on='id', how='left')[
        ['name', 'brand', 'positive_sentiment_percent']].drop_duplicates()





