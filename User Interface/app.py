import streamlit as st
import requests
from PIL import Image
import io
import re
import tempfile

API_URL = "http://127.0.0.1:8000"

# Fonction pour récupérer la liste des images disponibles
def get_image_list():
    try:
        response = requests.get(f"{API_URL}/images", timeout=10)
        response.raise_for_status()
        return response.json()["images"]
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des images : {e}")
        return []

# Fonction pour afficher une image
def display_image(image_id):
    try:
        response = requests.get(f"{API_URL}/images/{image_id}", timeout=10)
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
        mask_response = requests.get(f"{API_URL}/masks/{mask_name}", timeout=10)
        mask_response.raise_for_status()
        mask = Image.open(io.BytesIO(mask_response.content))
        return mask
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération du masque : {e}")
        return None

# Fonction pour effectuer une prédiction
def make_prediction(image):
    try:
        # Sauvegarder l'image dans un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_image:
            image.save(temp_image.name)
            temp_image.seek(0)
            files = {"file": (temp_image.name, open(temp_image.name, 'rb'), "image/png")}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
            response.raise_for_status()

            # Récupérer l'image de la prédiction
            prediction_image = Image.open(io.BytesIO(response.content))
            return prediction_image
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None

# Configuration de l'application
st.title("Application de Prédiction d'Images")

# Récupérer la liste des images disponibles
image_list = get_image_list()

# Afficher la liste des images disponibles
st.sidebar.header("Liste des Images Disponibles")
selected_image = st.sidebar.selectbox("Choisissez une image", image_list)

# Afficher l'image sélectionnée et son masque
if selected_image:
    image = display_image(selected_image)
    if image:
        st.image(image, caption='Image sélectionnée', use_column_width=True)

        # Bouton pour afficher le masque
        if st.button("Afficher le masque"):
            mask_name = get_mask_name(selected_image)
            st.write(f"Nom du masque recherché : {mask_name}")  # Affiche le nom du masque recherché
            mask = display_mask(mask_name)
            if mask:
                st.image(mask, caption='Masque correspondant', use_column_width=True)
            else:
                st.write("Masque non disponible pour cette image.")

        # Effectuer une prédiction
        if st.button("Effectuer une prédiction"):
            prediction_image = make_prediction(image)
            if prediction_image:
                # Redimensionner la prédiction pour qu'elle soit de la même taille que l'image de base
                prediction_image = prediction_image.resize(image.size)
                st.image(prediction_image, caption='Prédiction', use_column_width=True)
