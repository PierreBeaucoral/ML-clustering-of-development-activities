# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:49:16 2024

@author: Pierre Beaucoral
"""


import os
from datetime import datetime
import pandas as pd
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
import pretty_errors

pretty_errors.configure(
    display_timestamp=1,
    timestamp_function=lambda: datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
)

# Function to import CSV
def load_csv(file_path, dtype=str, delimiter=","):
    print("Loading CSV file:", file_path)
    return pd.read_csv(file_path, encoding='utf-8', dtype=dtype, low_memory=False, delimiter=delimiter).dropna(subset=['raw_text'])

def replace_entities_with_whitespace(text, regions):
    for entity_type in regions:
        text = text.replace(entity_type, ' ')
    return text

if __name__ == "__main__":
    start_time = datetime.now()

    # Set the working directory
    os.chdir('\your\directory')

    # Load CSV file
    df = load_csv(os.path.join('Data/Dataset.csv'), dtype={'raw_text': str, "PurposeCode": float})

    # Load SpaCy NLP model
    nlp = spacy.load("en_core_web_trf")
    regions = ['GPE', 'LOC', "ORG", "NORP", "PERS", "EVENT", "DATE", "LANGUAGE", "LAW", "CARDINAL"]

    # Redact entities from 'raw_text' column
    df['raw_text'] = df['raw_text'].apply(lambda text: replace_entities_with_whitespace(text, regions))

    # Print information about the loaded DataFrame
    print("Loaded CSV file. DataFrame shape:", df.shape)

    # Assign a unique category to each TEXT
    df['label'] = 'project' + df['raw_text'].astype(str).astype('category').cat.codes.astype(str)
    df.to_csv("Data/labeled_projects.csv", index=False, header=True, sep=',')

    # Drop duplicates in project descriptions
    df = df.drop_duplicates(subset='label', keep='first')
    print("DataFrame shape after dropping duplicates:", df.shape)
    
    # Create a directory to save visualizations if it doesn't exist
    output_dir = os.path.join(os.getcwd(), 'visualizationsretreated')
    os.makedirs(output_dir, exist_ok=True)
    
    # Assuming 'raw_text' and 'year' are the columns you want to use for topic modeling and time visualization
    docs = df['raw_text'].tolist()
    timestamps = df['Year'].tolist()  # Replace 'Year' with your actual timestamp column

    # Selection of your setup for the different steps of BERTopic
    ## select the model you want to use for embeddings
    sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    ## Realize the embeddings
    embeddings = sentence_model.encode(docs, show_progress_bar=True)

    ## Setting your hbdscan model 
    hdbscan_model = HDBSCAN(min_cluster_size=500, metric='euclidean', cluster_selection_method='leaf', prediction_data=True)
        
    ## Setting your hbdscan model 
    umap_model = UMAP(n_neighbors=100, n_components=12,min_dist=0, metric='cosine')

    ## Setting parameter for c-TF-IDF scores
    representation_model = MaximalMarginalRelevance(diversity=0.3)
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=10)

    # Topic modeling using BERTopic with your custom model
    topic_model = BERTopic( hdbscan_model=hdbscan_model, umap_model=umap_model, representation_model=representation_model,language="multilingual")
    topics, _ = topic_model.fit_transform(docs,embeddings)
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    
    #Save your model with embeddings (quite demanding)
    topic_model.save("\your\repo")

    # Evaluate using different metricsP

    ## Generate `X` and `labels` only for non-outlier topics (as they are technically not clusters)
    umap_embeddings = topic_model.umap_model.transform(embeddings)
    indices = [index for index, topic in enumerate(topics) if topic != -1]
    X = umap_embeddings[np.array(indices)]
    labels = [topic for index, topic in enumerate(topics) if topic != -1]
    
    ## Outliers info
    
    outliers_info = topic_model.get_topic_info(-1)
    outliers_count = outliers_info['Count']
    print("Number of outliers:", outliers_count[0])

    ## Calculate different clustering quality scores
    silhouette = silhouette_score(X, labels)
    print("Silhouette Score:", silhouette)
    davies_bouldin = davies_bouldin_score(X, labels)
    print("davies_bouldin Score:", davies_bouldin)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    print("calinski_harabasz Score:", calinski_harabasz)
    
    # Reduce outliers 
  
    ## Use the "c-TF-IDF" strategy with a threshold
    new_topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=0.6)
    
    ## Generate `X` and `labels` only for non-outlier topics (as they are technically not clusters)
    umap_embeddings = topic_model.umap_model.transform(embeddings)
    indices = [index for index, topic in enumerate(new_topics) if topic != -1]
    X = umap_embeddings[np.array(indices)]
    labels = [topic for index, topic in enumerate(new_topics) if topic != -1]
    
    ## Outliers info
    
    outliers_info = topic_model.get_topic_info(-1)
    outliers_count1 = np.sum(np.array(new_topics) == -1)
    print("Number of outliers:", outliers_count1)

    ## Calculate different clustering quality scores
    silhouette1 = silhouette_score(X, labels)
    print("Silhouette Score after first reducing:", silhouette1)
    davies_bouldin1 = davies_bouldin_score(X, labels)
    print("davies_bouldin Score after first reducing:", davies_bouldin1)
    calinski_harabasz1 = calinski_harabasz_score(X, labels)
    print("calinski_harabasz Score after first reducing:", calinski_harabasz1)
    
    ## Reduce remaining outliers with the "embeddings" strategy (more computation time than previous)
    new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings, threshold=0.75)
    
    ## Generate `X` and `labels` only for non-outlier topics (as they are technically not clusters)
    umap_embeddings = topic_model.umap_model.transform(embeddings)
    indices = [index for index, topic in enumerate(new_topics) if topic != -1]
    X = umap_embeddings[np.array(indices)]
    labels = [topic for index, topic in enumerate(new_topics) if topic != -1]
    
    ## Outliers info
    
    outliers_info = topic_model.get_topic_info(-1)
    outliers_count2 = np.sum(np.array(new_topics) == -1)
    print("Number of outliers:", outliers_count2)

    ## Calculate different clustering quality scores
    silhouette2 = silhouette_score(X, labels)
    print("Silhouette Score after second reducing:", silhouette2)
    davies_bouldin2 = davies_bouldin_score(X, labels)
    print("davies_bouldin Score after second reducing:", davies_bouldin2)
    calinski_harabasz2 = calinski_harabasz_score(X, labels)
    print("calinski_harabasz Score after second reducing:", calinski_harabasz2)
    

    ## Update topics with the reduced outliers
    topic_model.update_topics(docs, topics=new_topics)
    
    # re-use the hdbscan to automaticaly reduce topics that will seem similar
    topic_model.reduce_topics(docs, nr_topics="auto")
    
    ## Outliers info
    
    outliers_info = topic_model.get_topic_info(-1)
    outliers_count3 = np.sum(np.array(new_topics) == -1)
    print("Number of outliers:", outliers_count3)

    ## Calculate different clustering quality scores
    silhouette3 = silhouette_score(X, labels)
    print("Silhouette Score after second reducing:", silhouette3)
    davies_bouldin3 = davies_bouldin_score(X, labels)
    print("davies_bouldin Score after second reducing:", davies_bouldin3)
    calinski_harabasz3 = calinski_harabasz_score(X, labels)
    print("calinski_harabasz Score after second reducing:", calinski_harabasz3)
    
    # Define the differents scores calculated in our process in a same object
    scores = {
    "Original": {
        "Silhouette Score": silhouette,
        "davies_bouldin Score": davies_bouldin,
        "calinski_harabasz Score": calinski_harabasz,
        "Nb. of outliers": outliers_count[0]
    },
    "After First Reduction": {
        "Silhouette Score": silhouette1,
        "davies_bouldin Score": davies_bouldin1,
        "calinski_harabasz Score": calinski_harabasz1,
        "Nb. of outliers": outliers_count1
    },
    "After Second Reduction": {
        "Silhouette Score": silhouette2,
        "davies_bouldin Score": davies_bouldin2,
        "calinski_harabasz Score": calinski_harabasz2,
        "Nb. of outliers": outliers_count2
    },
    "After Topic Reduction": {
        "Silhouette Score": silhouette3,
        "davies_bouldin Score": davies_bouldin3,
        "calinski_harabasz Score": calinski_harabasz3,
        "Nb. of outliers": outliers_count3
    }
}
    ## Convert the scores dictionary to a DataFrame
    scores_df = pd.DataFrame(scores)
    
    ## Save results
    scores_df.to_csv("scores.csv", index=False, header=True, sep=',')
    
    # Add the updated topics to your DataFrame
    df['Topic'] = topics

    # saving the differents output 
    ## Save the dataframe of unique projects with their associated topic in csv
    df.to_csv("projects_clusters.csv", index=False, header=True, sep=',')
    
    ## Save csv dataframe with topic info
    Topic_info = topic_model.get_topic_info()
    Topic_info.to_csv("topic_info.csv", index=False, header=True, sep=',')

    
    # Visualizations of Topics

    ## Distribution of topics
    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html(os.path.join(output_dir, "topics_visualization.html"))
    
    ## Get the total number of unique topics
    total_topics = len(set(topics))
    
    ## Set a reasonable number of clusters based on your data
    n_clusters = min(250, (total_topics)-10)

    ## Visualize Topic Similarity
    fig_heatmap = topic_model.visualize_heatmap(n_clusters=n_clusters)  # Set a reasonable number of clusters based on your data
    fig_heatmap.write_html(os.path.join(output_dir, "heatmap_visualization.html"))
    
    ## Visualize Term Score Decline
    fig_term_rank = topic_model.visualize_term_rank()
    fig_term_rank.write_html(os.path.join(output_dir, "term_rank_visualization.html"))
    
    ## Visualize Topics over Time using the 'Year' variable
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    fig_over_time = topic_model.visualize_topics_over_time(topics_over_time)
    fig_over_time.write_html(os.path.join(output_dir, "topics_over_time_visualization.html"))
    
    ## Visualize Topics per DonorName
    donors = df['DonorName']  # Replace 'DonorName' with your actual donor column
    topics_per_donor = topic_model.topics_per_class(docs=docs, classes=donors)
    fig_per_donor = topic_model.visualize_topics_per_class(topics_per_donor, top_n_topics=total_topics, normalize_frequency=False, custom_labels=False, title='<b>Projects per donors</b>', width=1250, height=900)
    fig_per_donor.write_html(os.path.join(output_dir, "topics_per_donor_visualization.html"))
    
    ## Visualize Hierarchy of Topics
    fig_hierarchy = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
    fig_hierarchy.write_html(os.path.join(output_dir, "hierarchy_visualization.html"))
    
    ## Visualize Topic Probability Distribution
    topics, probabilities = topic_model.fit_transform(docs)  # Re-run fit_transform to get topics and probabilities
    fig_distribution = topic_model.visualize_distribution(probabilities)
    fig_distribution.write_html(os.path.join(output_dir, "distribution_visualization.html"))

    ## Assignment of each raw text in clusters 
    ### Perform dimensionality reduction with UMAP    
    reduced_embeddings = UMAP(n_neighbors=100, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    ### Creation of the plotly
    fig_assignement = topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings, hide_document_hover=True)
    fig_assignement.write_html(os.path.join(output_dir, "fig_assignement.html"))
    
    ## word scores per topic 
    fig_word = topic_model.visualize_barchart(top_n_topics=total_topics)
    fig_word.write_html(os.path.join(output_dir, "fig_word.html"))

    # Merge topic with the original global dataset  
    ## Load the 'labeled_projects' DataFrame
    labeled_projects = load_csv(os.path.join('Data/labeled_projects.csv'), dtype={'PurposeCode': float})
    
    ## Merge DataFrames on the 'label' column
    merged_df = pd.merge(labeled_projects, df[['label', 'Topic']], on='label', how='left')

    ## Save the merged DataFrame
    merged_df.to_csv("merged_projects.csv", index=False, header=True, sep=',')

    print("Merged DataFrame saved as 'merged_projects.csv'.")

    print("Visualizations completed.")
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time)/60
    print(f"Script executed in {elapsed_time.total_seconds():.2f} minutes.")
