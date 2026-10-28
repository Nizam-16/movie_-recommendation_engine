import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Page configuration
st.set_page_config(page_title="Movie Recommendation Engine", page_icon="🎬", layout="centered")

@st.cache_data
def load_data_and_model():
    """Loads the dataset and computes the similarity matrix."""
    data = {
        'Title': [
            'Inception', 'The Matrix', 'Interstellar', 'Toy Story', 
            'Finding Nemo', 'The Dark Knight', 'Gladiator', 'The Lord of the Rings',
            'Avatar', 'Titanic', 'Jurassic Park', 'Spider-Man: Into the Spider-Verse',
            'The Godfather', 'Pulp Fiction', 'Whiplash'
        ],
        'Genre': [
            'Sci-Fi Action Thriller', 'Sci-Fi Action', 'Sci-Fi Adventure Drama', 'Animation Family Comedy',
            'Animation Family Adventure', 'Action Crime Drama', 'Action Adventure Drama', 'Fantasy Adventure',
            'Sci-Fi Action Adventure', 'Romance Drama', 'Sci-Fi Adventure Thriller', 'Animation Action Sci-Fi',
            'Crime Drama', 'Crime Drama', 'Drama Music'
        ],
        'Description': [
            'A thief who steals corporate secrets through dream sharing technology.',
            'A computer hacker learns about the true nature of reality and his role in the war against its controllers.',
            'A team of explorers travel through a wormhole in space in an attempt to ensure humanity survival.',
            'A cowboy doll is profoundly threatened and jealous when a new spaceman action figure supplants him.',
            'After his son is captured in the Great Barrier Reef, a timid clownfish sets out on a journey to bring him home.',
            'When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests.',
            'A former Roman general sets out to exact vengeance against the corrupt emperor who murdered his family.',
            'A meek hobbit of the Shire and eight companions set out on a journey to destroy the powerful One Ring.',
            'A paraplegic marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world.',
            'A seventeen-year-old aristocrat falls in love with a kind, poor artist aboard the luxurious, ill-fated R.M.S. Titanic.',
            'During a preview tour, a theme park suffers a major power breakdown, allowing cloned dinosaurs to run loose.',
            'Teen Miles Morales becomes the Spider-Man of his universe, and must join with five spider-powered individuals from other dimensions.',
            'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant youngest son.',
            'The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
            'A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing.'
        ]
    }
    df = pd.DataFrame(data)
    df['Combined_Features'] = df['Genre'] + ' ' + df['Description']
    
    # Vectorize and compute similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['Combined_Features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, cosine_sim

# Load cached data and model
df, cosine_sim = load_data_and_model()

# UI Layout
st.title("🎬 Movie Recommendation Engine")
st.markdown("Select a movie you love, and our machine learning model will suggest similar titles based on genres and plot descriptions.")

# Dropdown for movie selection
selected_movie = st.selectbox("Choose a movie from the list:", df['Title'].values)

# Slider to choose number of recommendations
num_recs = st.slider("Number of recommendations:", min_value=1, max_value=5, value=3)

# Button to trigger recommendation logic
if st.button("Get Recommendations", type="primary"):
    matched_rows = df[df['Title'].str.lower() == selected_movie.lower()]
    
    if not matched_rows.empty:
        idx = matched_rows.index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recs + 1]
        
        st.subheader(f"Because you liked **{selected_movie}**, you might enjoy:")
        
        for i, score in sim_scores:
            rec_title = df['Title'].iloc[i]
            rec_genre = df['Genre'].iloc[i]
            rec_desc = df['Description'].iloc[i]
            rec_score = round(score * 100, 1)
            
            # Display each recommendation in a clean card format
            with st.container(border=True):
                st.markdown(f"### 🎥 {rec_title}")
                st.markdown(f"**Genre:** {rec_genre}")
                st.markdown(f"**Plot:** {rec_desc}")
                st.progress(score, text=f"Match Score: {rec_score}%")