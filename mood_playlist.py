from transformers import pipeline
import sys

# Load pre-trained emotion detection model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Pre-defined playlists (song titles for demo purposes)
playlists = {
    "positive": ["Happy by Pharrell Williams", "Sweet Caroline", "Dancing Queen by ABBA"],
    "negative": ["Creep by Radiohead", "Hurt by Johnny Cash", "Tears in Heaven by Eric Clapton"],
    "anger": ["Sweet Child O' Mine by Guns N' Roses", "Break Stuff by Limp Bizkit", "Killing in the Name by Rage Against the Machine"],
    "sadness": ["Someone Like You by Adele", "Yesterday by The Beatles", "The Night We Met by Lord Huron"],
    "neutral": ["Bohemian Rhapsody by Queen", "Imagine by John Lennon", "Hotel California by Eagles"]
}

# Check for input
if len(sys.argv) != 2:
    print("Usage: python mood_playlist.py \"your text here\"")
    sys.exit(1)

text = sys.argv[1]

# Detect mood
result = emotion_classifier(text)[0][0]  # Get top emotion
mood = result['label'].lower()
score = result['score']

# Map mood to playlist
if mood in ["joy", "positive"]:
    selected_playlist = playlists["positive"]
    mood_name = "Happy"
elif mood in ["sadness"]:
    selected_playlist = playlists["sadness"]
    mood_name = "Sad"
elif mood in ["anger"]:
    selected_playlist = playlists["anger"]
    mood_name = "Angry"
elif mood in ["fear", "disgust", "negative"]:
    selected_playlist = playlists["negative"]
    mood_name = "Stressed/Negative"
else:
    selected_playlist = playlists["neutral"]
    mood_name = "Neutral"

# Output result
print(f"Input Text: {text}")
print(f"Detected Mood: {mood_name} (Confidence: {score:.2f})")
print("Suggested Playlist:")
for song in selected_playlist:
    print(f"- {song}")