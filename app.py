import pickle
import streamlit as st
import requests
import os
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def generate_models():
    """Generate model files if they don't exist"""
    if not os.path.exists('model/movie_list.pkl') or not os.path.exists('model/similarity.pkl'):
        st.info("🔧 Generating recommendation models... This may take a few minutes on first run.")
        
        with st.spinner("Loading and processing movie data..."):
            # Load datasets
            movies = pd.read_csv('tmdb_5000_movies.csv')
            credits = pd.read_csv('tmdb_5000_credits.csv')
            
            # Merge datasets
            movies = movies.merge(credits, on='title')
            
            # Data processing functions
            def convert(text):
                L = []
                for i in ast.literal_eval(text):
                    L.append(i['name']) 
                return L

            def convert3(text):
                L = []
                counter = 0
                for i in ast.literal_eval(text):
                    if counter < 3:
                        L.append(i['name'])
                    counter+=1
                return L

            def fetch_director(text):
                L = []
                for i in ast.literal_eval(text):
                    if i['job'] == 'Director':
                        L.append(i['name'])
                return L

            def collapse(L):
                L1 = []
                for i in L:
                    L1.append(i.replace(' ',''))
                return L1

            # Clean data
            movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
            movies.dropna(inplace=True)

            # Process features
            movies['genres'] = movies['genres'].apply(convert)
            movies['keywords'] = movies['keywords'].apply(convert)
            movies['cast'] = movies['cast'].apply(convert3)
            movies['crew'] = movies['crew'].apply(fetch_director)

            # Remove spaces
            movies['cast'] = movies['cast'].apply(collapse)
            movies['crew'] = movies['crew'].apply(collapse)
            movies['genres'] = movies['genres'].apply(collapse)
            movies['keywords'] = movies['keywords'].apply(collapse)

            # Process overview
            movies['overview'] = movies['overview'].apply(lambda x:x.split())

            # Create tags
            movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

            # Create final dataset
            new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
            new['tags'] = new['tags'].apply(lambda x: ' '.join(x))

        with st.spinner("Creating similarity matrix..."):
            # Vectorize and calculate similarity
            cv = CountVectorizer(max_features=5000,stop_words='english')
            vector = cv.fit_transform(new['tags']).toarray()
            similarity = cosine_similarity(vector)

        with st.spinner("Saving model files..."):
            # Save files
            os.makedirs('model', exist_ok=True)
            pickle.dump(new, open('model/movie_list.pkl', 'wb'))
            pickle.dump(similarity, open('model/similarity.pkl', 'wb'))
            
        st.success("✅ Models generated successfully! Reloading page...")
        st.rerun()

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url).json()
    
    # Check if poster_path exists
    poster_path = data.get('poster_path')
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    else:
        return "https://via.placeholder.com/500x750.png?text=No+Image"  # Default placeholder

def get_movie_details(movie_title):
    """Get detailed information about a movie from the tags"""
    try:
        movie_info = movies[movies['title'] == movie_title].iloc[0]
        tags = movie_info['tags']
        
        # Extract overview (first part of tags before genre keywords)
        overview_part = tags.split(' Action')[0] if ' Action' in tags else tags.split(' Adventure')[0] if ' Adventure' in tags else tags.split(' Drama')[0] if ' Drama' in tags else tags[:200]
        
        # Extract common genre keywords from tags
        genres = []
        genre_keywords = ['Action', 'Adventure', 'Drama', 'Comedy', 'Horror', 'Thriller', 'Romance', 'ScienceFiction', 'Fantasy', 'Crime', 'Mystery', 'Animation']
        for genre in genre_keywords:
            if genre in tags:
                genres.append(genre)
        
        # Extract actor names (capitalized words that look like names)
        import re
        potential_actors = re.findall(r'\b[A-Z][a-z]+[A-Z][a-z]+\b', tags)
        actors = potential_actors[:5]  # Take first 5 potential actor names
        
        return {
            'title': movie_info['title'],
            'movie_id': movie_info['movie_id'],
            'overview': overview_part if len(overview_part) > 20 else "Plot details included in movie features.",
            'genres': genres[:5],
            'cast': actors,
            'crew': potential_actors[-2:] if len(potential_actors) > 5 else potential_actors[:2],  # Assume last ones might be crew
            'keywords': tags
        }
    except Exception as e:
        return {
            'title': movie_title,
            'movie_id': 0,
            'overview': "Movie details not available",
            'genres': [],
            'cast': [],
            'crew': [],
            'keywords': ""
        }

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    recommended_movies = []
    
    for i in distances[1:6]:  # Top 5 recommended movies
        movie_data = movies.iloc[i[0]]
        recommended_movies.append({
            'title': movie_data.title,
            'movie_id': movie_data.movie_id,
            'similarity_score': round(i[1] * 100, 1),
            'poster': fetch_poster(movie_data.movie_id)
        })

    return recommended_movies

# Streamlit UI
st.header('🎬 Movie Recommender System')
st.subheader('Find movies similar to what you love!')

# Generate models if they don't exist
generate_models()

movies = pickle.load(open('model/movie_list.pkl', 'rb'))
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

movie_list = movies['title'].values
st.write(f"🎭 **{len(movie_list)} movies available**")

# Better selectbox with instructions
selected_movie = st.selectbox(
    "🔍 Search and select a movie:",
    movie_list,
    index=0,
    help="Type to search or scroll to find your movie"
)

st.write(f"**Selected Movie:** {selected_movie}")

# Show details of selected movie
selected_details = get_movie_details(selected_movie)
if selected_details:
    with st.expander("📋 Show Selected Movie Details"):
        col_poster, col_details = st.columns([1, 2])
        with col_poster:
            st.image(fetch_poster(selected_details['movie_id']), width=200)
        with col_details:
            st.write(f"**📖 Overview:** {selected_details['overview']}")
            if selected_details['genres']:
                st.write(f"**🎭 Genres:** {', '.join(selected_details['genres'])}")
            if selected_details['cast']:
                st.write(f"**⭐ Cast:** {', '.join(selected_details['cast'])}")
            if selected_details['crew']:
                st.write(f"**🎬 Crew:** {', '.join(selected_details['crew'])}")

# Show all movie features in a separate section
with st.expander("🏷️ All Movie Features Used for Similarity"):
    if selected_details:
        st.text(selected_details['keywords'][:500] + "..." if len(selected_details['keywords']) > 500 else selected_details['keywords'])

# Automatically show recommendations when a movie is selected
st.subheader('🎯 Recommended Movies Similar to ' + selected_movie)

with st.spinner('Finding similar movies...'):
    recommended_movies = recommend(selected_movie)

# Display recommendations with detailed info
for i, movie in enumerate(recommended_movies):
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.image(movie['poster'], width=150)
        
        with col2:
            st.subheader(f"{i+1}. {movie['title']}")
            st.write(f"**🎯 Similarity Score:** {movie['similarity_score']}%")
            
            # Get details for this recommended movie
            movie_details = get_movie_details(movie['title'])
            if movie_details and movie_details['overview'] != "Movie details not available":
                st.write(f"**📖 Plot:** {movie_details['overview'][:150]}..." if len(movie_details['overview']) > 150 else movie_details['overview'])
                if movie_details['genres']:
                    st.write(f"**🎭 Genres:** {', '.join(movie_details['genres'])}")
                if movie_details['cast']:
                    st.write(f"**⭐ Main Cast:** {', '.join(movie_details['cast'][:3])}")
                if movie_details['crew']:
                    st.write(f"**🎬 Crew:** {', '.join(movie_details['crew'][:2])}")
            else:
                st.write("📋 Movie features are used for similarity matching")
        
        st.divider()
