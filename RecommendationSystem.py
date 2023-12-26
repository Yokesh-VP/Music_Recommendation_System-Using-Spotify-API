import requests
import base64
import pandas as pd
import spotipy
import datetime
import streamlit as st
import math
from streamlit_lottie import st_lottie
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

client_id = "d0afd59a307e4570bfd7921d8dec89de"
client_secret = "3a5850f76b444ec5a8b74ce5142189e5"
client_access = "{}:{}".format(client_id,client_secret)

client_access_base64 = base64.b64encode(client_access.encode())


headers = {
    'Authorization': 'Basic {}'.format(client_access_base64.decode())
}
data = {
    'grant_type': 'client_credentials'
}
url = 'https://accounts.spotify.com/api/token'

response = requests.post(url,data = data,headers=headers)

if response.status_code == 200:
    access_token = response.json()['access_token']
    print("Access token obtained successfully.")
else:
    print("Error obtaining access token.")
    exit()


def playlist_info(playlist_id,access_token):
    # Set up Spotipy with the access token
    sp = spotipy.Spotify(auth=access_token)
    # Get the tracks from the playlist
    playlist_tracks = sp.playlist_tracks(playlist_id,fields='items(track(id, name, artists, album(id, name, images)))')
    # Extract relevant information and store in a list of dictionaries
    music_data=[]
    for tracks in playlist_tracks["items"]:
        track_info = tracks["track"]
        track_id = track_info["id"]
        track_name = track_info["name"]
        album_id = track_info["album"]["id"]
        album_name = track_info["album"]["name"]
        artists = ",".join(artist["name"] for artist in track_info['artists'])
        # Get audio features for the track
        audio_features = sp.audio_features(track_id)[0] if track_id != "Not available" else None
        # Get release date of the album
        try:
            album_info = sp.album(album_id) if album_id != "Not available" else None
            release_date = album_info["release_date"] if album_info else None
        except:
            release_date = None
        # Get popularity of the track
        try:
            track_info = sp.track(track_id) if track_id != "Not available" else None
            popularity = track_info["popularity"] if track_info else None
        except:
            popularity = None

        # Add additional track information to the track data
        track_data ={
            "Track ID" : track_id,
            "Track Name" : track_name,
            "Album ID" : album_id,
            "Album Name" : album_name,
            "Artists" : artists,
            'Explicit': track_info.get('explicit', None),
            'External URLs': track_info.get('external_urls', {}).get('spotify', None),
            "Release Date":release_date,
            "Popularity":popularity,
            "Image URL":track_info["album"]["images"][0]["url"] if track_info["album"]["images"] else None,
            "Duration (MS)" : audio_features["duration_ms"] if audio_features else None,
            "Danceability" : audio_features["danceability"] if audio_features else None,
            "Energy" : audio_features["energy"] if audio_features else None,
            "Key" : audio_features["key"] if audio_features else None,
            "Loudness": audio_features["loudness"] if audio_features else None,
            "Mode" : audio_features["mode"] if audio_features else None,
            "Speechiness" : audio_features["speechiness"] if audio_features else None,
            "Acousticness" : audio_features["acousticness"] if audio_features else None,
            "Instrumentalness": audio_features["instrumentalness"] if audio_features else None,
            "Liveness": audio_features["liveness"] if audio_features else None,
            "Valence": audio_features["valence"] if audio_features else None,
            "Tempo": audio_features["tempo"] if audio_features else None
        }
        music_data.append(track_data)
    df = pd.DataFrame(music_data)
    df.dropna(inplace=True)
    return df


def calculated_popularity(release_date):
    if len(release_date) == 4:
        time_span = datetime.date.today().year - int(release_date)
        weight = 1 / (((time_span*365)+time_span//4)+1)
        return weight
    release_date = datetime.datetime.strptime(release_date, '%Y-%m-%d')
    time_span = datetime.datetime.now() - release_date
    weight = 1 / (time_span.days+1)
    return weight

def scaling(music_df):               
    scaler = MinMaxScaler()
    audio_features = music_df[['Danceability', 'Energy', 'Key', 'Loudness', 'Mode','Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness','Valence', 'Tempo']]
    scaled_audio_features = scaler.fit_transform(audio_features)
    return scaled_audio_features


def recommendations_audio_features(music_df,input_song_name,num_recommendations):
    scaled_audio_features = scaling(music_df)
    filt = music_df["Track Name"] == input_song_name
    input_song_index = music_df[filt].index[0] ## [0] bcz it return index in the form of list 
    similarity = cosine_similarity([scaled_audio_features[input_song_index]],scaled_audio_features) ### [] bcz we need 2 d array instead of 1 d array if we use eithout it one 1d and other is 2d
    similar_song_indices = similarity.argsort()[0][::-1][1:math.ceil(num_recommendations/2)+1] ### [1:num+1] bcz the input song index is also included ,[0] bcz index is in [[]] so [0]
    similar_songs = music_df.iloc[similar_song_indices]
    return similar_songs[['Track Name', 'Album Name', 'Artists', 'Release Date', 'Popularity','Image URL']]


def text_similarity(music_df,input_song_name,num_recommendations):
    song_vectorizer = CountVectorizer()
    song_vectorizer.fit(music_df["Track Name"])
    filt = music_df["Track Name"] == input_song_name 
    text_array1 = song_vectorizer.transform(music_df.loc[filt,"Track Name"]).toarray()
    text_array2 = song_vectorizer.transform(music_df["Track Name"])
    score = cosine_similarity(text_array1,text_array2)
    similar_song_indices = score.argsort()[0][::-1][1:math.floor(num_recommendations/2)+1] ### [1:num+1] bcz the input song index is also included ,[0] bcz index is in [[]] so [0]
    similar_songs = music_df.iloc[similar_song_indices]
    return similar_songs[['Track Name', 'Album Name', 'Artists', 'Release Date', 'Popularity','Image URL']]

def hybrid_recommendations(playlist_id,input_song_name,num_recommendations):
    music_df = playlist_info(playlist_id,access_token)
    if input_song_name not in music_df['Track Name'].values:
        print("The specified song is not in the playlist\nBut Here are the top most popular songs you might like From the Playlist")
        return music_df[['Track Name', 'Album Name', 'Artists', 'Release Date', 'Popularity','Image URL']].sort_values(by="Popularity",ascending=False).head()
    else:
        audio_rec = recommendations_audio_features(music_df,input_song_name,num_recommendations)
        text_rec = text_similarity(music_df,input_song_name,num_recommendations)        
        popularity = music_df.loc[music_df['Track Name'] == input_song_name, 'Popularity'].values[0]
        weighted_calculated_popularity = popularity * calculated_popularity(music_df.loc[music_df['Track Name'] == input_song_name, 'Release Date'].values[0])
        weight_rec = pd.DataFrame({'Track Name':input_song_name, 
            'Album Name': music_df.loc[music_df['Track Name'] == input_song_name, 'Album Name'].values[0], 
            'Artists': music_df.loc[music_df['Track Name'] == input_song_name, 'Artists'].values[0], 
            'Release Date':music_df.loc[music_df['Track Name'] == input_song_name, 'Release Date'].values[0], 
            'Popularity': weighted_calculated_popularity,
            'Image URL':music_df.loc[music_df['Track Name'] == input_song_name, 'Image URL'].values[0] 
             }, index=[0])
        hybrid_recommendations = pd.concat([audio_rec,text_rec,weight_rec],ignore_index=True)
        hybrid_recommendations = hybrid_recommendations.sort_values(by="Popularity",ascending=False)
        hybrid_recommendations = hybrid_recommendations[hybrid_recommendations["Track Name"] != input_song_name]
        return hybrid_recommendations

def Web():
    def load_music_url(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_music=load_music_url("https://lottie.host/41ff735f-52fc-4aa5-bff6-c2b9606ae267/WMfSSoQV0c.json")
    st_lottie(lottie_music, speed=1, height=300, key="initial")
    st.title("Music Recommendation System Using Spotify API")
    st.markdown("Our recommendations help you find audio you’ll enjoy, whether that’s an old favorite  or a new song")
    st.markdown("Our personalized recommendations are tailored to your unique taste, taking into account a variety of factors, such as similar audio features and similar track names")
    st_lottie(load_music_url("https://lottie.host/2adee15d-118b-4ffe-82f6-bd5790efe396/fGuHaH4yra.json"), speed=1, height=100)
    playlist_id = st.text_input("**Enter the Spotify Playlist ID Link**")
    if st.button("Get Playlist"):
        playlist_id = playlist_id.split("/")[4][:22]
        music_df = playlist_info(playlist_id,access_token)
        st.write(music_df)
    st.markdown("**TO GET SONG RECOMMENDATION.** 👇")
    input_song_name = st.text_input("**Enter the Song Name to get recommended from the playlist**")
    num_recommendations = st.slider('**Number of Recommendations**', min_value=1, max_value=10, value=5, step=1)
    if st.button("Get Recommendations"):
        recommendations = hybrid_recommendations(playlist_id,input_song_name,num_recommendations)
        st.write("Recommended Songs Are:")
        col1, mid, col2 = st.columns([1,1,20])
        for index,track in recommendations.iterrows():  
            with col1:
                track_image = track['Image URL']
                st.image(track_image, width = 55)
            with col2:
                st.write(track["Track Name"])
                st.write("")
                st.write("")
if __name__ == "__main__":
    Web()
