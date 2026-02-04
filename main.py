
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#Cargar el dataset
df = pd.read_csv("top_100_spotify_songs_2025.csv")
df.head()
df.info()
df.describe()

#Análisis Exploratorio
top_artists = df['Artist'].value_counts().head(10)

plt.figure()
top_artists.plot(kind='bar')
plt.title("Top 10 artistas con más canciones en el top 100")
plt.xlabel("Artista")
plt.ylabel("Número de canciones")
plt.show()


y = df['Popularity_Score']
features = [
    'Rank',
    'Spotify_Streams_Millions',
    'Duration_Seconds',
    'Explicit'
]

X = df[features]

X.info()
df['Explicit'] = df['Explicit'].map({
    'Yes': 1,
    'No': 0
})
df['Streams_per_Second'] = df['Spotify_Streams_Millions'] / df['Duration_Seconds']
features.append('Streams_per_Second')
X = df[features]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2F}")
print(f"R2 Score: {r2:.2f}")

importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

importance

plt.figure()
sns.barplot(data=importance, x='Coefficient', y='Feature')
plt.title("Impacto de las variables en la popularidad")
plt.show()
