import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ========================
# 1. Charger dataset
# =======================
df = pd.read_csv(r"C:\Users\hp\Downloads\Car details v3.csv")
df = df.dropna()

# Nettoyage colonnes
#df["mileage"] = df["mileage"].str.replace(" kmpl", "").astype(float)
df["engine"] = df["engine"].str.replace(" CC", "").astype(float)
df["max_power"] = df["max_power"].str.replace(" bhp", "").astype(float)

# Ajouter des colonnes fictives pour localisation (si dataset réel pas dispo)
if 'latitude' not in df.columns:
    np.random.seed(42)
    df['latitude'] = np.random.uniform(33.5, 34.5, size=len(df))
    df['longitude'] = np.random.uniform(-7.8, -6.5, size=len(df))

# Encodage
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("selling_price", axis=1)
y = df_encoded["selling_price"]

# =========================
# 2. Modèle ML
# =========================
model = RandomForestRegressor()
model.fit(X, y)

# =========================
# 3. Clustering
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# =========================
# 4. Interface Streamlit
# =========================
st.set_page_config(page_title="🚗 Car Prediction Dashboard ", layout="wide")

st.title("🚗 Car Prediction Dashboard ")
st.markdown("### Dashboard interactif pour explorer, prédire et analyser les voitures")

menu = st.sidebar.radio(" Menu",[
    "📊 Dashboard",
    "🤖 Prédiction",
    "🧠 Clustering",
    "🔍 Infos voiture",
    "🗺️ Carte interactive"
])

# =========================
# 5. Dashboard et filtres avancés
# =========================
if menu == "📊 Dashboard":
    st.subheader("Filtres avancés")

    min_price, max_price = st.slider("Prix (€)", int(df.selling_price.min()), int(df.selling_price.max()),
                                     (int(df.selling_price.min()), int(df.selling_price.max())))
    min_year, max_year = st.slider("Année", int(df.year.min()), int(df.year.max()),
                                   (int(df.year.min()), int(df.year.max())))
    fuel_type = st.multiselect("Carburant", df["fuel"].unique(), default=df["fuel"].unique())

    filtered_df = df[(df["selling_price"] >= min_price) &
                     (df["selling_price"] <= max_price) &
                     (df["year"] >= min_year) &
                     (df["year"] <= max_year) &
                     (df["fuel"].isin(fuel_type))]

    st.dataframe(filtered_df.head(20))

    st.subheader("Histogramme des prix")
    fig, ax = plt.subplots()
    ax.hist(filtered_df["selling_price"], bins=20, color="skyblue", edgecolor="black")
    st.pyplot(fig)

    st.subheader("Prix vs Année")
    fig2, ax2 = plt.subplots()
    ax2.scatter(filtered_df["year"], filtered_df["selling_price"], c=filtered_df["cluster"], cmap="Set1")
    st.pyplot(fig2)

# =========================
# 6. Prediction
# =========================
elif menu == "🤖 Prédiction":
    st.subheader("Prédire le prix d'une voiture")

    year = st.slider("Année", int(df.year.min()), int(df.year.max()))
    kms = st.slider("Kilometers Driven", 0, int(df["km_driven"].max()))
    engine = st.slider("Engine CC", int(df.engine.min()), int(df.engine.max()))

    if st.button("Predict"):
        input_data = np.zeros((1, X.shape[1]))
        input_data[0][0] = year
        input_data[0][1] = kms
        input_data[0][2] = engine
        prediction = model.predict(input_data)
        st.success(f"Prix estimé : {round(prediction[0], 2)} €")

# =========================
# 7. Clustering
# =========================
elif menu == "🧠 Clustering":
    st.subheader("Visualisation des clusters")

    fig, ax = plt.subplots()
    scatter = ax.scatter(df["engine"], df["selling_price"], c=df["cluster"], cmap="Set1")
    st.pyplot(fig)

    st.write("Nombre de voitures par cluster")
    st.bar_chart(df["cluster"].value_counts())

# =========================
# 8. Infos voiture
# =========================
elif menu == "🔍 Infos voiture":
    st.subheader("Rechercher une voiture")

    car_name = st.selectbox("Choisir voiture", df["name"].unique())
    car_data = df[df["name"] == car_name]
    st.dataframe(car_data)

# =========================
# 9. Carte interactive
# =========================
elif menu == "🗺️ Carte interactive":
    st.subheader("Carte interactive des voitures")

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        initial_view_state=pdk.ViewState(
            latitude=33.9,
            longitude=-7.15,
            zoom=8,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position='[longitude, latitude]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
                pickable=True
            )
        ]
    ))
