import pandas as pd
quora = pd.read_csv('./udemy_repo/NLP_Natural_Language_Processing_with_Python/05-Topic-Modeling/quora_questions.csv')

print(quora.head())


# # Preprocessing
# #### Task: Use TF-IDF Vectorization to create a vectorized document term matrix. You may want to explore the max_df and min_df parameters.
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(quora['Question'])
print(dtm)


# # Non-negative Matrix Factorization
# 
# #### TASK: Using Scikit-Learn create an instance of NMF with 20 expected components. (Use random_state=42)..
from sklearn.decomposition import NMF
nmf_model = NMF(n_components=20, random_state=42)
nmf_model.fit(dtm)


# #### TASK: Print our the top 15 most common words for each of the 20 topics.
for index, topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# #### TASK: Add a new column to the original quora dataframe that labels each question into one of the 20 topic categories.
print(quora.head())
topic_results = nmf_model.transform(dtm)

topic_results.argmax(axis=1)

quora['Topic'] = topic_results.argmax(axis=1)

print(quora.head(10))


# # Great job!
