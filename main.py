from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import difflib

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('dataset.csv')

# Remove exact duplicates
df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first')

# Numeric features
features = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity']

# Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# Fit the KNN model
knn = NearestNeighbors(n_neighbors=20, algorithm='auto')
knn.fit(scaled_data)

# Find closest match for a given song name
def find_closest_song(song_name):
    possible_matches = difflib.get_close_matches(song_name, df['track_name'].str.lower(), n=1, cutoff=0.6)
    return possible_matches[0] if possible_matches else None

# Recommendation function
def recommend_song(song_name):
    closest_match = find_closest_song(song_name)
    if closest_match is None:
        return []
    
    song = df[df['track_name'].str.lower() == closest_match].sort_values(by='popularity', ascending=False).head(1)
    
    if song.empty:
        return []
    
    song_features = song[features].values.reshape(1, -1)
    scaled_song = scaler.transform(song_features)
    
    distances, indices = knn.kneighbors(scaled_song)
    
    recommended_songs = df.iloc[indices[0]]
    recommended_songs = recommended_songs[recommended_songs['track_genre'] == song['track_genre'].values[0]]
    
    recommended_songs['similarity_score'] = 1.0
    recommended_songs.loc[recommended_songs['artists'] == song['artists'].values[0], 'similarity_score'] *= 1.2
    recommended_songs = recommended_songs.sort_values(by=['similarity_score', 'popularity'], ascending=[False, False])
    
    # Ensure the searched song comes first
    recommended_songs = pd.concat([song, recommended_songs]).drop_duplicates().head(5)
    
    return recommended_songs[['track_name', 'artists', 'track_genre', 'popularity']].to_dict(orient='records')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    song_name = request.json.get('song_name', '')
    recommendations = recommend_song(song_name)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
