"""
data_generation.py

This module generates synthetic sports commentary data for training our abuse detection model.
It creates realistic examples of both abusive and non-abusive content, incorporating various
linguistic patterns found in social media sports discussions.
"""


import pandas as pd
import numpy as np
import random
import string
from datetime import datetime

def generate_simple_tweet():
    """
    Generate a basic tweet with clear abusive or non-abusive content.
    Returns both the tweet text and its label.
    """
    teams = ["Liverpool", "Chelsea", "Arsenal", "Manchester United", 
             "Manchester City", "Tottenham", "Newcastle", "Aston Villa"]
    players = ["Salah", "Kane", "Haaland", "Saka", "Bruno", "Odegaard", 
              "Son", "Foden", "Rashford"]
    
    positive_templates = [
        "Brilliant performance by {player} today! Pure class 👏",
        "{team} showing great spirit in this match! Keep it up 💪",
        "What a goal by {player}! Absolutely world class ⚽",
        "So proud of {team} today, fantastic team effort!",
        "The passion from {player} is incredible to watch!"
    ]
    
    abusive_templates = [
        "Get out of our club {player}, absolute fraud! 🤬",
        "{team} are pathetic, worst team I've ever seen",
        "{player} is completely useless, should never play again",
        "Disgraceful performance by {team}, bunch of overpaid clowns",
        "Hope {player} never plays again, complete waste of space"
    ]
    
    is_abusive = random.choice([True, False])
    template = random.choice(abusive_templates if is_abusive else positive_templates)
    text = template.format(
        team=random.choice(teams),
        player=random.choice(players)
    )
    
    return text, is_abusive

def add_typos(text, probability=0.3):
    """Add realistic typing errors to text."""
    if random.random() > probability:
        return text
        
    chars = list(text)
    error_type = random.choice(['swap', 'miss', 'repeat'])
    
    if error_type == 'swap' and len(chars) > 2:
        idx = random.randint(0, len(chars)-2)
        chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    elif error_type == 'miss' and len(chars) > 3:
        idx = random.randint(0, len(chars)-1)
        chars.pop(idx)
    elif error_type == 'repeat' and len(chars) > 1:
        idx = random.randint(0, len(chars)-1)
        chars.insert(idx, chars[idx])
        
    return ''.join(chars)

def generate_realistic_tweets(n_samples=10000):
    """Generate a dataset of realistic tweets with various complexities."""
    tweets_data = []
    
    for _ in range(n_samples):
        # Generate base tweet
        text, is_abusive = generate_simple_tweet()
        
        # Add complexity with 30% probability
        if random.random() < 0.3:
            text = add_typos(text)
        
        # Add sarcasm with 20% probability
        if random.random() < 0.2:
            sarcastic_markers = ["Yeah right", "Sure", "Obviously", "Clearly"]
            text = f"{random.choice(sarcastic_markers)}, {text.lower()}"
            # Sarcasm might flip the abuse label
            is_abusive = not is_abusive if random.random() < 0.5 else is_abusive
        
        tweets_data.append({
            'text': text,
            'label': int(is_abusive)
        })
    
    return pd.DataFrame(tweets_data)

# Generate the dataset
print("Generating dataset...")
df = generate_realistic_tweets(10000)

# Add timestamp to filename for versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'synthetic_tweets_{timestamp}.csv'

# Save the dataset
df.to_csv(filename, index=False)
print(f"\nDataset saved as: {filename}")

# Display dataset statistics
print("\nDataset Statistics:")
print(f"Total samples: {len(df)}")
print("\nLabel distribution:")
print(df['label'].value_counts(normalize=True).multiply(100).round(2))

# Display some examples
print("\nExample tweets:")
for _, row in df.sample(5).iterrows():
    print(f"\nText: {row['text']}")
    print(f"Label: {'Abusive' if row['label'] else 'Non-abusive'}")
