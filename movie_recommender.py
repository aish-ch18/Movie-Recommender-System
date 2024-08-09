import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Custom movie dataset
movies_data = {
    'title': [
        'The Godfather', 'The Dark Knight', 'Pulp Fiction',
        'The Shawshank Redemption', 'The Matrix', 'Forrest Gump',
        'Inception', 'Fight Club', 'The Silence of the Lambs', 'Se7en'
    ],
    'genres': [
        'Crime Drama', 'Action Crime Drama', 'Crime Drama',
        'Drama', 'Action Sci-Fi', 'Drama Romance',
        'Action Adventure Sci-Fi', 'Drama', 'Crime Drama Thriller', 'Crime Drama Thriller'
    ],
    'director': [
        'Francis Ford Coppola', 'Christopher Nolan', 'Quentin Tarantino',
        'Frank Darabont', 'Lana Wachowski Lilly Wachowski', 'Robert Zemeckis',
        'Christopher Nolan', 'David Fincher', 'Jonathan Demme', 'David Fincher'
    ],
    'actors': [
        'Marlon Brando Al Pacino James Caan', 'Christian Bale Heath Ledger Aaron Eckhart',
        'John Travolta Uma Thurman Samuel L. Jackson', 'Tim Robbins Morgan Freeman',
        'Keanu Reeves Laurence Fishburne', 'Tom Hanks Robin Wright Gary Sinise',
        'Leonardo DiCaprio Joseph Gordon-Levitt Ellen Page', 'Brad Pitt Edward Norton',
        'Jodie Foster Anthony Hopkins Lawrence A. Bonney', 'Brad Pitt Morgan Freeman'
    ],
    'plot': [
        'The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.',
        'When the menace known as the Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham.',
        'The lives of two mob hitmen, a boxer, a gangster\'s wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
        'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
        'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
        'The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75.',
        'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.',
        'An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into much more.',
        'A young F.B.I. cadet must confide in an incarcerated and manipulative killer to receive his help on catching another serial killer.',
        'Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives.'
    ]
}

# Create DataFrame
movies = pd.DataFrame(movies_data)

# Combine relevant features into a single string
movies['soup'] = movies['genres'] + ' ' + movies['director'] + ' ' + movies['actors'] + ' ' + movies['plot']

# Create the TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['soup'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Test the recommendation system
print(get_recommendations('The Godfather'))
print(get_recommendations('The Matrix'))
