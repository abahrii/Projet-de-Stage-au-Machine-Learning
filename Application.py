import streamlit as st
import pickle
import pandas as pd

st.title("Interface de Prédiction pour Benchmarking Énergétique")

# Sélection de la cible via la barre latérale
target_choice = st.sidebar.selectbox("Sélectionnez la cible", ["data_TCEnergy", "data_EC2"])

# Chargement du modèle correspondant
if target_choice == "data_TCEnergy":
    with open("best_model_data_TCEnergy.pkl", "rb") as f:
        model = pickle.load(f)
    st.header("Prédiction pour data_TCEnergy")
else:
    with open("best_model_data_EC2.pkl", "rb") as f:
        model = pickle.load(f)
    st.header("Prédiction pour data_EC2")

# Choix du mode d'entrée : saisie manuelle ou chargement d'un fichier CSV
option = st.radio("Mode d'entrée", ["Saisie manuelle", "Charger un fichier CSV"])

if option == "Saisie manuelle":
    st.write("Entrez les valeurs des caractéristiques séparées par une virgule (ex: 12.3, 45.6, 78.9, ...)")
    input_str = st.text_input("Valeurs des caractéristiques")
    if st.button("Prédire"):
        try:
            # Conversion de la chaîne en liste de flottants
            input_list = [float(x.strip()) for x in input_str.split(",")]
            # Conversion en DataFrame (une seule ligne)
            data_input = pd.DataFrame([input_list])
            prediction = model.predict(data_input)
            st.write("Prédiction :", prediction)
        except Exception as e:
            st.error(f"Erreur dans l'entrée : {e}")
else:
    uploaded_file = st.file_uploader("Charger un fichier CSV contenant les caractéristiques", type=["csv"])
    if uploaded_file is not None:
        data_input = pd.read_csv(uploaded_file)
        st.write("Aperçu des données chargées :")
        st.dataframe(data_input)
        if st.button("Prédire"):
            try:
                prediction = model.predict(data_input)
                st.write("Prédictions :")
                st.write(prediction)
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

