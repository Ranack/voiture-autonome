import os
import subprocess
import sys
import logging
from io import BytesIO

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vérifier et installer les bibliothèques nécessaires
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import tensorflow as tf
    os.environ["SM_FRAMEWORK"] = "tf.keras"
    import segmentation_models as sm
except ImportError:
    logger.info("Installation des packages requis...")
    install_package("tensorflow==2.18.0")
    install_package("segmentation-models==1.0.1")
    import tensorflow as tf
    os.environ["SM_FRAMEWORK"] = "tf.keras"
    import segmentation_models as sm

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from PIL import Image
import numpy as np

app = FastAPI()

# Définition des chemins
dirs = {
    "images": "../Images/Photos",
    "masks": "../Images/Mask",
    "model": "../Model/efficientnet_fpn_model_best_iou_diceloss.keras"
}

# Vérifier si le modèle existe
if not os.path.exists(dirs["model"]):
    raise ValueError(f"Modèle introuvable : {dirs['model']}")

# Chargement du modèle avec gestion d'erreur
try:
    model = tf.keras.models.load_model(dirs["model"], compile=False)
    logger.info("Modèle chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle : {e}")
    raise ValueError("Impossible de charger le modèle.")

# Palette de couleurs pour la visualisation (exemple pour 8 classes)
color_palette = np.array([
    [0, 0, 0],       # Fond (noir)
    [255, 0, 0],     # Classe 1 (rouge)
    [0, 255, 0],     # Classe 2 (vert)
    [0, 0, 255],     # Classe 3 (bleu)
    [255, 255, 0],   # Classe 4 (jaune)
    [0, 255, 255],   # Classe 5 (cyan)
    [255, 0, 255],   # Classe 6 (magenta)
    [255, 255, 255]  # Classe 7 (blanc)
], dtype=np.uint8)

# Fonction pour récupérer la liste des images disponibles
def get_image_list():
    try:
        return [f for f in os.listdir(dirs["images"]) if f.endswith(('.png', '.jpg', '.jpeg'))]
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des images : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

# Fonction pour effectuer une prédiction avec le modèle
def make_prediction(image):
    try:
        logger.info("Prédiction en cours...")

        # Redimensionner correctement l'image pour le modèle
        height, width = 1024, 512  # Correspondance au modèle
        image = image.resize((width, height))  # (512, 1024)

        # Vérification des dimensions de l'image
        logger.info(f"Dimensions de l'image après redimensionnement : {image.size}")

        # Convertir l'image en tableau numpy et normaliser
        image_array = np.array(image).astype(np.float32) / 255.0
        if image_array.shape[-1] != 3:  # Vérifie que l'image est bien RGB
            raise ValueError("L'image doit avoir 3 canaux (RGB).")

        # Ajouter une dimension batch
        image_array = np.expand_dims(image_array, axis=0)

        # Vérification des dimensions finales
        logger.info(f"Dimensions de l'image après conversion en tableau : {image_array.shape}")

        # Prédiction
        prediction = model.predict(image_array)
        logger.info("Prédiction terminée.")

        # Vérification de la forme de la prédiction
        logger.info(f"Dimensions de la prédiction : {prediction.shape}")

        # Cas pour une segmentation multi-classe (8 canaux)
        if prediction.shape[-1] > 1:  # Si la prédiction a plus d'un canal (multi-classe)
            # Appliquer argmax pour obtenir la classe prédite pour chaque pixel
            prediction = np.argmax(prediction, axis=-1)  # Shape (1, height, width)
            prediction = np.squeeze(prediction)  # Shape (height, width)
        else:
            # Si la prédiction n'a qu'un seul canal, pas besoin d'argmax
            prediction = prediction.squeeze()  # Supprimer les dimensions inutiles

        # Convertir la prédiction en image colorée
        prediction_image = color_palette[prediction]
        prediction_image = Image.fromarray(prediction_image.astype(np.uint8))
        return prediction_image
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.get("/health")
async def health_check():
    """ Vérifie si l'API est fonctionnelle. """
    return JSONResponse(content={"status": "API is running"})

@app.get("/images")
async def list_images():
    return JSONResponse(content={"images": get_image_list()})

@app.get("/images/{image_id}")
async def get_image(image_id: str):
    image_path = os.path.join(dirs["images"], image_id)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image non trouvée")
    return FileResponse(image_path)

@app.get("/masks/{mask_id}")
async def get_mask(mask_id: str):
    mask_path = os.path.join(dirs["masks"], mask_id)
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Masque non trouvé")
    return FileResponse(mask_path)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        prediction_image = make_prediction(image)

        # Sauvegarder l'image de la prédiction dans un buffer
        buffer = BytesIO()
        prediction_image.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/png")
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")
