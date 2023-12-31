import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from tmdbv3api import TMDb
from tmdbv3api import Movie


st.set_page_config(
    page_title="CineWave",
    page_icon=":🎞️:",
    layout="wide",  
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .signature {
        font-family: 'Bodoni MT Black', serif;
        font-size: 15px;
        color: #ffffff; /* Couleur du texte */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialisation de l'API TMDB
tmdb = TMDb()
tmdb_movie = Movie()
# Charger les variables d'environnement depuis le fichier .env
load_dotenv()
tmdb.api_key = os.getenv('TMDB_API_KEY')

### Titre de l'application
st.title("🎥🎞️CineMatch🎞️🎥: Votre système de recommendation de films")
st.markdown('<p>Développé par <span class="signature">AlMiDEV</span></p>', unsafe_allow_html=True)

# Chargement des données depuis le fichier CSV
def load_data():
    data = pd.read_csv('movies2023.csv')
    return data

# Chargement des données
df = load_data()

# Supprimer les lignes avec des valeurs manquantes
df.dropna(inplace=True)
# Réinitialiser les index
df.reset_index(drop=True, inplace=True)
# Supprimer les duplicatas dans la colonne 'MovieID'
df.drop_duplicates(subset='MovieID', keep='first', inplace=True)
# Réinitialiser les index
df.reset_index(drop=True, inplace=True)

# Création d'une nouvelle colonne 'Description' avec la concaténation des colonnes pertinentes
df['Description'] = df['Overview'] + ' ' + df['Actors'] + ' ' + df['Directors'] + ' ' + df['Genres']
# Sélection des colonnes nécessaires pour le nouveau DataFrame
new_df = df[['MovieID', 'Title', 'Description']].copy()

def get_movie_titles(indices):
    return df.iloc[indices]['Title'].tolist()

# Charger la matrice de similarité depuis le fichier
article_similarity = np.load('articlesimilarity.npy')
description_similarity = np.load('descriptionsimilarity.npy')

option = st.selectbox("Sélectionnez le type de recherche", ("Genre", "Acteur", "Titre", "Réalisateur"))

def get_recommendations_based_on_genre(genre, article_similarity, num_recommendations=100):
    # Vérifier si la valeur est une chaîne de caractères avant de diviser
    df['Genres'] = df['Genres'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])
    
    # Trouver les films similaires en fonction du genre donné
    genre_indices = df[df['Genres'].apply(lambda x: genre in x)]
    recommendations = genre_indices.index.tolist()
    return recommendations[:num_recommendations]

def get_recommendations_based_on_actor(actor, article_similarity, num_recommendations=50):
    # Vérifier si la valeur est une chaîne de caractères avant de diviser
    df['Actors'] = df['Actors'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

    # Trouver les films similaires en fonction de l'acteur donné
    actor_indices = df[df['Actors'].apply(lambda x: actor in x)]
    recommendations = actor_indices.index.tolist()
    return recommendations[:num_recommendations]

def get_recommendations_based_on_director(director, article_similarity, num_recommendations=50):
    # Vérifier si la valeur est une chaîne de caractères avant de diviser
    df['Directors'] = df['Directors'].apply(lambda x: x.split(', ') if isinstance(x, str) else [])

    # Trouver les films similaires en fonction du réalisateur donné
    director_indices = df[df['Directors'].apply(lambda x: director in x)]
    recommendations = director_indices.index.tolist()
    return recommendations[:num_recommendations]

def get_movie_recommendations_by_title(movie_title, df, description_similarity, num_recommendations=100):
    # Trouver l'index du film dans le DataFrame
    movie_index = df[df['Title'] == movie_title].index[0]
    # Obtenir les indices des films similaires pour le film donné
    similar_movies = sorted(list(enumerate(description_similarity[movie_index])), key=lambda x: x[1], reverse=True)
    # Récupérer les titres des films recommandés
    recommended_movie_indices = [movie[0] for movie in similar_movies[1:num_recommendations+1]]  # Exclut le film lui-même
    recommended_movies = df.iloc[recommended_movie_indices]['Title'].tolist()
    
    return recommended_movies



if option == "Genre":
    # Interface pour la recherche par genre
    genre_unique = [''] + df['Genres'].str.strip().str.split(',').explode().str.strip().unique().tolist()
    genre_input = st.selectbox("Veuillez sélectionner un genre", genre_unique)
    if st.button("Rechercher"):
        if genre_input:
            recommendations_genre = get_recommendations_based_on_genre(genre_input, article_similarity)
            for movie_index in recommendations_genre:
                movie_info = df.iloc[movie_index]
                st.title(movie_info['Title'])  # Afficher le titre du film
                st.write(f"Année de sortie : {int(movie_info['ReleaseYear'])}")  # Afficher l'année de sortie
                st.write(f"Réalisateur : {movie_info['Directors']}")  # Afficher le réalisateur
                actors_list = movie_info['Actors'].split(', ')[:10]  # Prendre les cinq premiers acteurs
                st.write(f"Acteurs principaux : {', '.join(actors_list)}")  # Afficher les acteurs principaux
                st.write(f"Genre : {', '.join(movie_info['Genres'])}")  # Afficher le genre
                movie = tmdb_movie.search(movie_info['Title'])[0]
                poster_path = movie.poster_path
                if poster_path:
                    image_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
                    st.image(image_url, caption=movie.title, width=200)  # Afficher l'image
                st.write(movie.overview)  # Afficher la description
        else:
            st.write("Veuillez sélectionner un genre pour effectuer la recherche.")

elif option == "Acteur":
    # Interface pour la recherche par acteur
    actor_unique = [''] + df['Actors'].str.strip().str.split(',').explode().str.strip().unique().tolist()
    actor_input = st.selectbox("Veuillez entrer le nom d'un acteur", actor_unique)
    if st.button("Rechercher"):
        if actor_input:
            recommendations_actor = get_recommendations_based_on_actor(actor_input, article_similarity)
            for movie_index in recommendations_actor:
                movie_info = df.iloc[movie_index]
                st.title(movie_info['Title'])  # Afficher le titre du film
                st.write(f"Année de sortie : {int(movie_info['ReleaseYear'])}")  # Afficher l'année de sortie
                st.write(f"Réalisateur : {', '.join(movie_info['Genres'])}")  # Afficher le réalisateur
                actors_list = movie_info['Actors']  # Prendre les cinq premiers acteurs
                actor=', '.join(actors_list).split(', ')[:10]
                st.write(f"Acteurs principaux : {', '.join(actor)}")  # Afficher les acteurs principaux
                st.write(f"Genre : {movie_info['Genres']}")  # Afficher le genre
                movie = tmdb_movie.search(movie_info['Title'])[0]
                poster_path = movie.poster_path
                if poster_path:
                    image_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
                    st.image(image_url, caption=movie.title, width=200)  # Afficher l'image
                st.write(movie.overview)  # Afficher la description
        else:
            st.write("Veuillez sélectionner un acteur pour effectuer la recherche.")

elif option == "Réalisateur":
    # Interface pour la recherche par réalisateur
    director_unique = [''] + df['Directors'].str.strip().str.split(',').explode().str.strip().unique().tolist()
    director_input = st.selectbox("Veuillez entrer le nom d'un réalisateur", director_unique)
    if st.button("Rechercher"):
        if director_input:
            recommendations_director = get_recommendations_based_on_director(director_input, article_similarity)
            for movie_index in recommendations_director:
                movie_info = df.iloc[movie_index]
                st.title(movie_info['Title'])  # Afficher le titre du film
                st.write(f"Année de sortie : {int(movie_info['ReleaseYear'])}")  # Afficher l'année de sortie
                st.write(f"Réalisateur : {', '.join(movie_info['Directors'])}")  # Afficher le réalisateur
                actors_list = movie_info['Actors'].split(', ')[:10]  # Prendre les cinq premiers acteurs
                st.write(f"Acteurs principaux : {', '.join(actors_list)}")  # Afficher les acteurs principaux
                st.write(f"Genre : {movie_info['Genres']}")  # Afficher le genre
                movie = tmdb_movie.search(movie_info['Title'])[0]
                poster_path = movie.poster_path
                if poster_path:
                    image_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
                    st.image(image_url, caption=movie.title, width=200)  # Afficher l'image
                st.write(movie.overview)  # Afficher la description
        else:
            st.write("Veuillez sélectionner un réalisateur pour effectuer la recherche.")

elif option == "Titre":
    # Interface pour la recherche par titre
    # Obtenir les titres uniques et ajouter une option vide au début de la liste
    titles_unique = [''] + df['Title'].unique().tolist()
    title_input = st.selectbox("Entrez le nom d'un film", titles_unique)
    if st.button("Rechercher"):
        if title_input:
            num_recommendations = 100  # Nombre de recommandations à générer
            recommendations = get_movie_recommendations_by_title(title_input, new_df, description_similarity, num_recommendations)
            for movie_title in recommendations:
                st.title(movie_title)  # Afficher le titre du film
                movie_info = df[df['Title'] == movie_title].iloc[0]
                st.write(f"Année de sortie : {int(movie_info['ReleaseYear'])}")  # Afficher l'année de sortie
                st.write(f"Réalisateur : {movie_info['Directors']}")  # Afficher le réalisateur
                actors_list = movie_info['Actors'].split(', ')[:10]  # Prendre les cinq premiers acteurs
                st.write(f"Acteurs principaux : {', '.join(actors_list)}")  # Afficher les acteurs principaux
                st.write(f"Genre : {movie_info['Genres']}")  # Afficher le genre
                movie = tmdb_movie.search(movie_title)[0]
                poster_path = movie.poster_path
                if poster_path:
                    image_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
                    st.image(image_url, caption=movie.title, width=200)  # Afficher l'image
                st.write(movie.overview)  # Afficher la description
        else:
            st.write("Veuillez sélectionner un film pour effectuer la recherche.")
        