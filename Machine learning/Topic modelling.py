# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:49:16 2024

@author: pibeauco
"""

import os
from datetime import datetime
import pandas as pd
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance, TextGeneration
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
import pretty_errors
from nltk.corpus import stopwords
from bertopic.vectorizers import ClassTfidfTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

pretty_errors.configure(
    display_timestamp=1,
    timestamp_function=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')
)

# Function to import CSV
def load_csv(file_path, dtype=str, delimiter=","):
    print("Loading CSV file:", file_path)
    return pd.read_csv(file_path, encoding='utf-8', dtype=dtype, low_memory=False, delimiter=delimiter).dropna(subset=['raw_text'])

# Function to create directories

def create_directories(output_dirs):
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)

# Function to evaluate clustering quality

def evaluate_clustering(umap_embeddings, labels):
    silhouette = silhouette_score(umap_embeddings, labels)
    davies_bouldin = davies_bouldin_score(umap_embeddings, labels)
    calinski_harabasz = calinski_harabasz_score(umap_embeddings, labels)
    return silhouette, davies_bouldin, calinski_harabasz

# Function to get the custom labels (here from zephyr)

def extract_zephyr_labels(topic_model):
    zephyr_topics = topic_model.get_topics(full=True)["Zephyr"]
    zephyr_labels = {}
    for topic_id, topic_info in zephyr_topics.items():
        # Assuming each label is in the first element of the list
        label = topic_info[0][0].strip()
        # Remove unwanted characters like brackets
        label = label.replace('[', '').replace(']', '').replace('"', '').replace('<|assistant|>', '').strip()
        if label:
            zephyr_labels[topic_id] = label
        else:
            zephyr_labels[topic_id] = f"Topic {topic_id}"  # Fallback label

    # Add label for outliers
    zephyr_labels[-1] = "Outlier Topic"
    return zephyr_labels

# Function to manualy modify costum label (if needed)

def update_dictionary_value(dictionary, target_key, new_value):
    """
    Update the value of a specific key in a dictionary.

    :param dictionary: The dictionary to update.
    :param target_key: The key whose value needs to be updated.
    :param new_value: The new value to set for the target key.
    :return: The updated dictionary.
    """
    if target_key in dictionary:
        dictionary[target_key] = new_value
    return dictionary

# Main function to run the entire workflow
def main():
    start_time = datetime.now()

    # Set the working directory
    os.chdir('cd/')

    # Load CSV file
    df = load_csv(os.path.join('Data/Dataset.csv'), dtype={'raw_text': str, "PurposeCode": float})

    print("Loaded CSV file. DataFrame shape:", df.shape)

    # Drop duplicates based on project descriptions
    df_unique = df.drop_duplicates(subset='raw_text', keep='first')
    df_unique.reset_index(drop=True, inplace=True)
    
    print("DataFrame shape after dropping duplicates:", df_unique.shape)
    
    # Create directories
    output_dir = os.path.join(os.getcwd(), 'visualizationszephyr')
    data_output_dir = os.path.join(os.getcwd(), 'dataframezephyr')
    create_directories([output_dir, data_output_dir])
    
    docs = df_unique['raw_text'].tolist()
    timestamps = df_unique['Year'].tolist()  # Replace 'Year' with your actual timestamp column

    #Settings of Bertopic
     
    final_stopwords_list = stopwords.words('english') + stopwords.words('french')+ stopwords.words('dutch')+ stopwords.words('spanish')

    ## Embeddings 
    sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=True)

    ## Clustering
    hdbscan_model = HDBSCAN(min_cluster_size=500, min_samples=400, metric='euclidean', cluster_selection_method='leaf', prediction_data=True)
    
    ## Dimensionality reduction
    umap_model = UMAP(n_neighbors=100, n_components=12, min_dist=0, metric='cosine')

    ## Labeling clusters
    representation_model = MaximalMarginalRelevance(diversity=0.3)
    final_stopwords_list = stopwords.words('english') + stopwords.words('german') + stopwords.words('french')+ stopwords.words('dutch')+ stopwords.words('spanish')
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words=final_stopwords_list, min_df=50, max_features=10000)
    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    model.eval()

    ### Pipeline
    generator = pipeline(
        model=model, tokenizer=tokenizer,
        task='text-generation',
        max_new_tokens=50,
        repetition_penalty=1.1
    )

    ### Zephyr prompt
    prompt = """You are a helpful, respectful and honest assistant for labeling topics coming from development project narrative descriptions..</s>

I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic, it should be in english and not exceed 10 words. Make sure you to only return the label and nothing more.</s>
<|assistant|>"""

    zephyr = TextGeneration(generator, prompt=prompt)
    representation_model = {"Zephyr": zephyr}

    # Topic modeling using BERTopic with your custom embedding model
    topic_model = BERTopic(hdbscan_model=hdbscan_model, umap_model=umap_model, representation_model=representation_model, language="multilingual", vectorizer_model=vectorizer_model, calculate_probabilities=True, low_memory=True)
    topics, probs = topic_model.fit_transform(docs, embeddings)
    hierarchical_topics = topic_model.hierarchical_topics(docs)
    
    print("Topic modeling completed.")

    # Clustering Evaluation and outlier reduction 

    umap_embeddings = topic_model.umap_model.transform(embeddings)
    indices = [index for index, topic in enumerate(topics) if topic != -1]
    X = umap_embeddings[np.array(indices)]
    labels = [topic for index, topic in enumerate(topics) if topic != -1]
    
    silhouette, davies_bouldin, calinski_harabasz = evaluate_clustering(X, labels)
    print("Silhouette Score:", silhouette)
    print("Davies-Bouldin Score:", davies_bouldin)
    print("Calinski-Harabasz Score:", calinski_harabasz)
    
    new_topics = topic_model.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=0.6)
    umap_embeddings = topic_model.umap_model.transform(embeddings)
    indices = [index for index, topic in enumerate(new_topics) if topic != -1]
    X = umap_embeddings[np.array(indices)]
    labels = [topic for index, topic in enumerate(new_topics) if topic != -1]
    
    silhouette1, davies_bouldin1, calinski_harabasz1 = evaluate_clustering(X, labels)
    print("Silhouette Score after first reducing:", silhouette1)
    print("Davies-Bouldin Score after first reducing:", davies_bouldin1)
    print("Calinski-Harabasz Score after first reducing:", calinski_harabasz1)
    
    new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings, threshold=0.725)
    umap_embeddings = topic_model.umap_model.transform(embeddings)
    indices = [index for index, topic in enumerate(new_topics) if topic != -1]
    X = umap_embeddings[np.array(indices)]
    labels = [topic for index, topic in enumerate(new_topics) if topic != -1]
    
    silhouette2, davies_bouldin2, calinski_harabasz2 = evaluate_clustering(X, labels)
    print("Silhouette Score after second reducing:", silhouette2)
    print("Davies-Bouldin Score after second reducing:", davies_bouldin2)
    print("Calinski-Harasz Score after second reducing:", calinski_harabasz2)
    
    topic_model.update_topics(docs, topics=new_topics)
    
    scores = {
        "Original": {
            "Silhouette Score": silhouette,
            "Davies-Bouldin Score": davies_bouldin,
            "Calinski-Harabasz Score": calinski_harabasz,
        },
        "After First Reduction": {
            "Silhouette Score": silhouette1,
            "Davies-Bouldin Score": davies_bouldin1,
            "Calinski-Harabasz Score": calinski_harabasz1,
        },
        "After Second Reduction": {
            "Silhouette Score": silhouette2,
            "Davies-Bouldin Score": davies_bouldin2,
            "Calinski-Harabasz Score": calinski_harabasz2,
        }
    }
    
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(os.path.join(data_output_dir, "scores.csv"), index=False, header=True, sep=',')
    
    # Extract labels correctly
    zephyr_labels = extract_zephyr_labels(topic_model)
    
    # Make same label shorter
    
    zephyr_labels = update_dictionary_value(zephyr_labels, 13, "Vocational training")
    zephyr_labels = update_dictionary_value(zephyr_labels, 20, "Integrated Rural Dev")
    zephyr_labels = update_dictionary_value(zephyr_labels, 76, "Refugees aid")
    zephyr_labels = update_dictionary_value(zephyr_labels, 167, "Salaries for civil society strengthening projects in Africa")
    zephyr_labels = update_dictionary_value(zephyr_labels, 182, "Eye health services")
    zephyr_labels = update_dictionary_value(zephyr_labels, 187, "Misean Cara dev project support")
    zephyr_labels = update_dictionary_value(zephyr_labels, 229, "Women's rights, Indigenous organizations, gender equity (Latin America)")
    zephyr_labels = update_dictionary_value(zephyr_labels, 290, "Air Pollution Mitigation")
    zephyr_labels = update_dictionary_value(zephyr_labels, 291, "AIDS Relief for Global Health")
    zephyr_labels = update_dictionary_value(zephyr_labels, 347, "Fertilizer Plant")
    zephyr_labels = update_dictionary_value(zephyr_labels, 351, "Programm 185, french cooperation and cultural action services")
    zephyr_labels = update_dictionary_value(zephyr_labels, 352, "Internship duties")
    zephyr_labels = update_dictionary_value(zephyr_labels, 357, "Fr-alliance-dev-employees")
    zephyr_labels = update_dictionary_value(zephyr_labels, 362, "Reception & Plac. Inside US (Migrants/Refugees)")
    zephyr_labels = update_dictionary_value(zephyr_labels, 365, "International french schools")
    zephyr_labels = update_dictionary_value(zephyr_labels, 367, "EIDHR CBSS Action Prog")
    zephyr_labels = update_dictionary_value(zephyr_labels, 371, "Cyclones project")
    zephyr_labels = update_dictionary_value(zephyr_labels, 373, "Nigerian Anti-corruption projects")
    zephyr_labels = update_dictionary_value(zephyr_labels, 379, "Cooperation with moldova")
    zephyr_labels = update_dictionary_value(zephyr_labels, 384, "Spain cultural centers cooperation")
    zephyr_labels = update_dictionary_value(zephyr_labels, 396, "national endowment for democracy grant")
    zephyr_labels = update_dictionary_value(zephyr_labels, 402, "integrated water resource management (Jordan, Palestine, Israel)")

    # Set the labels to the topic model
    topic_model.set_topic_labels(zephyr_labels)
    
    # Save some results as csv files
    doc_info = topic_model.get_document_info(docs)
    doc_info_df = pd.DataFrame(doc_info, columns=['Document', 'Topic'])
    
    df_unique['raw_text'] = df_unique['raw_text'].astype(str)
    doc_info_df['Document'] = doc_info_df['Document'].astype(str)
    merged_df = pd.merge(df_unique, doc_info_df, left_on='raw_text', right_on='Document', how='left')
    merged_df.drop(columns=['Document'], inplace=True)
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(os.path.join(data_output_dir, "topic_info.csv"), index=False, header=True, sep=',')
    merged_df = pd.merge(merged_df, topic_info, left_on='Topic', right_on='Topic', how='left')
    merged_df.to_csv(os.path.join(data_output_dir, "projects_clusters.csv"), index=False, header=True, sep=',')
    
    # Visualisation of topics 
     fig_topics = topic_model.visualize_topics(custom_labels=True)
    fig_topics.write_html(os.path.join(output_dir, "topics_visualization.html"))
    
    total_topics = len(set(topics))
    n_clusters = min(400, (total_topics) - 10)
    fig_heatmap = topic_model.visualize_heatmap(n_clusters=n_clusters, custom_labels=True)
    fig_heatmap.write_html(os.path.join(output_dir, "heatmap_visualization.html"))
    
    fig_term_rank = topic_model.visualize_term_rank(custom_labels=True)
    fig_term_rank.write_html(os.path.join(output_dir, "term_rank_visualization.html"))
    
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    fig_over_time = topic_model.visualize_topics_over_time(topics_over_time, custom_labels=True)
    fig_over_time.write_html(os.path.join(output_dir, "topics_over_time_visualization.html"))
    
    donors = df_unique['DonorName']
    topics_per_donor = topic_model.topics_per_class(docs=docs, classes=donors)
    fig_per_donor = topic_model.visualize_topics_per_class(topics_per_donor, top_n_topics=total_topics, normalize_frequency=False, custom_labels=True, title='<b>Projects per donors</b>', width=1250, height=900)
    fig_per_donor.write_html(os.path.join(output_dir, "topics_per_donor_visualization.html"))
    
    fig_hierarchy = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics, custom_labels=True)
    fig_hierarchy.write_html(os.path.join(output_dir, "hierarchy_visualization.html"))
    
    fig_distribution = topic_model.visualize_distribution(probabilities=probs)
    fig_distribution.write_html(os.path.join(output_dir, "distribution_visualization.html"))
    
    fig_word = topic_model.visualize_barchart(top_n_topics=total_topics, custom_labels=True)
    fig_word.write_html(os.path.join(output_dir, "fig_word.html"))
    
    reduced_embeddings = UMAP(n_neighbors=100, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    fig_assignement = topic_model.visualize_documents(docs,custom_labels=True, reduced_embeddings=reduced_embeddings, hide_document_hover=True)
    fig_assignement.write_html(os.path.join(output_dir, "fig_assignement.html"))
    fig_assignement_datamap = topic_model.visualize_document_datamap(docs,custom_labels=True, reduced_embeddings=reduced_embeddings, hide_document_hover=True)
    fig_assignement_datamap.write_html(os.path.join(output_dir, "fig_assignement_datamap.html"))
    
    all_merged_df = pd.merge(df, df_unique[['raw_text', 'Topic']], on='raw_text', how='left')
    all_merged_df.to_csv(os.path.join(data_output_dir, "merged_projects.csv"), index=False, header=True, sep=',')
    
    print("Merged DataFrame saved as 'merged_projects.csv'.")
    print("Visualizations completed. Saving model")
    
    topic_model.save("./my_model")
    
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds() / 60
    print(f"Script executed in {elapsed_time:.2f} minutes.")

if __name__ == "__main__":
    main()
