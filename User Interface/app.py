import streamlit as st
import requests
from PIL import Image
import io
import re
import tempfile

API_URL = "https://twisentiment-v2.azurewebsites.net"

# Fonction pour récupérer la liste des images disponibles
def get_image_list():
    try:
        response = requests.get(f"{API_URL}/images", timeout=60)
        response.raise_for_status()
        return response.json()["images"]
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des images : {e}")
        return []

# Fonction pour afficher une image
def display_image(image_id):
    try:
        response = requests.get(f"{API_URL}/images/{image_id}", timeout=60)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération de l'image : {e}")
        return None

# Fonction pour obtenir le nom du masque correspondant
def get_mask_name(image_name):
    base_name = re.sub(r'_leftImg8bit\.png', '', image_name)
    mask_name = f"{base_name}_gtFine_labelIds.png"
    return mask_name

# Fonction pour afficher le masque correspondant
def display_mask(mask_name):
    try:
        mask_response = requests.get(f"{API_URL}/masks/{mask_name}", timeout=60)
        mask_response.raise_for_status()
        mask = Image.open(io.BytesIO(mask_response.content))
        return mask
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération du masque : {e}")
        return None

# Fonction pour effectuer une prédiction
def make_prediction(image):
    try:
        with st.spinner('Prédiction en cours, veuillez patienter...'):
            # Sauvegarder l'image dans un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
                image.save(temp_image.name)
                temp_image.seek(0)
                files = {"file": (temp_image.name, open(temp_image.name, 'rb'), "image/png")}
                response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
                response.raise_for_status()

                # Récupérer l'image de la prédiction
                prediction_image = Image.open(io.BytesIO(response.content))
                return prediction_image
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None

# Configuration de l'application
st.title("Prédiction d'Images pour voitures autonomes")

# Récupérer la liste des images disponibles
image_list = get_image_list()

# Afficher le nombre total d'images disponibles
st.sidebar.write(f"Nombre total d'images disponibles : {len(image_list)}")

# Afficher la liste des images disponibles
st.sidebar.header("Liste des Images Disponibles")
selected_image = st.sidebar.selectbox("Choisissez une image", image_list)

# Initialiser l'état des boutons
if 'show_mask' not in st.session_state:
    st.session_state.show_mask = False
if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False

# Afficher l'image sélectionnée, le masque et la prédiction
if selected_image:
    image = display_image(selected_image)
    if image:
        st.image(image, caption='Image sélectionnée', use_column_width=True)

        # Bouton pour afficher le masque
        if st.button("Afficher le masque", key="mask_button"):
            st.session_state.show_mask = True

        # Bouton pour effectuer une prédiction
        if st.button("Effectuer une prédiction", key="prediction_button"):
            st.session_state.show_prediction = True

        # Afficher le masque si le bouton a été cliqué
        if st.session_state.show_mask:
            mask_name = get_mask_name(selected_image)
            mask = display_mask(mask_name)
            if mask:
                st.image(mask, caption='Masque correspondant', use_column_width=True)
            else:
                st.write("Masque non disponible pour cette image.")

        # Afficher la prédiction si le bouton a été cliqué
        if st.session_state.show_prediction:
            prediction_image = make_prediction(image)
            if prediction_image:
                # Redimensionner la prédiction pour qu'elle soit de la même taille que l'image de base
                prediction_image = prediction_image.resize(image.size)
                st.image(prediction_image, caption='Prédiction', use_column_width=True)
