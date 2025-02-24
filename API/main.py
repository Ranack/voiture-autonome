import os
import subprocess
import sys
import logging
from io import BytesIO
from collections import namedtuple
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from PIL import Image
import numpy as np

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

app = FastAPI()

# Définition des chemins
DIRS = {
    "images": os.path.join(os.path.dirname(__file__), '..', 'Images', 'Photos'),
    "masks": os.path.join(os.path.dirname(__file__), '..', 'Images', 'Mask'),
    "model": os.path.join(os.path.dirname(__file__), 'Model', 'efficientnet_fpn_model_best_iou_diceloss.keras')
}

# Vérifier si le modèle existe
if not os.path.exists(DIRS["model"]):
    raise ValueError(f"Modèle introuvable : {DIRS['model']}")

# Chargement du modèle avec gestion d'erreur
try:
    model = tf.keras.models.load_model(DIRS["model"], compile=False)
    logger.info("Modèle chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement modèle : {e}")
    raise ValueError("Impossible de charger le modèle.")

# Préchauffage du modèle
def warmup_model():
    try:
        logger.info("Préchauffage du modèle en cours...")
        # Créer une image fictive pour le préchauffage
        dummy_image = np.zeros((1, 1024, 512, 3), dtype=np.float32)
        model.predict(dummy_image)
        logger.info("Préchauffage du modèle terminé.")
    except Exception as e:
        logger.error(f"Erreur lors du préchauffage du modèle : {e}")

# Appeler la fonction de préchauffage
warmup_model()

# Définition de la structure Label
Label = namedtuple('Label', [
    'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval', 'color'
])

# Liste complète des labels
LABELS = [
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
]

# Dictionnaire pour regrouper les catId à des noms de catégories simplifiées
CATEGORY_MAPPING = {
    0: 'void',
    1: 'flat',
    2: 'construction',
    3: 'object',
    4: 'nature',
    5: 'sky',
    6: 'human',
    7: 'vehicle',
}

# Fonction pour mapper les catId aux catégories simplifiées
def map_category(label):
    new_category = CATEGORY_MAPPING.get(label.categoryId, 'unknown')
    return Label(
        name=label.name,
        id=label.id,
        trainId=label.trainId,
        category=new_category,
        categoryId=label.categoryId,
        hasInstances=label.hasInstances,
        ignoreInEval=label.ignoreInEval,
        color=label.color
    )

# Application de la fonction avec une compréhension de liste
GROUPED_LABELS = [map_category(label) for label in LABELS]

# Palette de couleurs pour la visualisation
COLOR_PALETTE = np.array([
    [0, 0, 0],         # Fond (noir)
    [230, 25, 75],     # Classe 1 (rouge foncé)
    [60, 180, 75],     # Classe 2 (vert)
    [255, 225, 25],    # Classe 3 (jaune clair)
    [0, 130, 200],     # Classe 4 (bleu)
    [245, 130, 48],    # Classe 5 (orange)
    [145, 30, 180],    # Classe 6 (violet)
    [240, 240, 240],   # Classe 7 (gris clair)
    [70, 240, 240],    # Classe 8 (cyan clair)
], dtype=np.uint8)

# Fonction pour récupérer la liste des images disponibles
def get_image_list():
    try:
        image_list = []
        for root, dirs, files in os.walk(DIRS["images"]):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Ajouter le chemin relatif du fichier
                    relative_path = os.path.relpath(os.path.join(root, file), DIRS["images"])
                    image_list.append(relative_path)
        logger.info(f"Images trouvées : {image_list}")
        return image_list
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des images : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

# Fonction pour effectuer une prédiction avec le modèle
def make_prediction(image):
    try:
        logger.info("Prédiction en cours...")

        # Redimensionner correctement l'image pour le modèle
        height, width = 1024, 512
        image = image.resize((width, height))

        # Convertir l'image en tableau numpy et normaliser
        image_array = np.array(image).astype(np.float32) / 255.0
        if image_array.shape[-1] != 3:
            raise ValueError("L'image doit avoir 3 canaux (RGB).")

        # Ajouter une dimension batch
        image_array = np.expand_dims(image_array, axis=0)

        # Prédiction
        prediction = model.predict(image_array)
        logger.info("Prédiction terminée.")

        # Cas pour une segmentation multi-classe
        if prediction.shape[-1] > 1:
            prediction = np.argmax(prediction, axis=-1).squeeze()
        else:
            prediction = prediction.squeeze()

        # Convertir la prédiction en image colorée
        prediction_image = Image.fromarray(COLOR_PALETTE[prediction].astype(np.uint8))
        return prediction_image
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.get("/health")
async def health_check():
    """ Vérifie si l'API est fonctionnelle. """
    return JSONResponse(content={"status": "API is running"})

@app.get("/status")
async def status():
    """ Vérifie le statut de l'API. """
    return JSONResponse(content={"status": "API is ready"})

@app.get("/images")
async def list_images():
    images = get_image_list()
    logger.info(f"Liste des images : {images}")
    return JSONResponse(content={"images": images})

@app.get("/images/{image_id}")
async def get_image(image_id: str):
    image_path = os.path.join(DIRS["images"], image_id)
    logger.info(f"Tentative d'accès à l'image : {image_path}")
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image non trouvée")
    return FileResponse(image_path)

@app.get("/masks/{mask_id}")
async def get_mask(mask_id: str):
    mask_path = os.path.join(DIRS["masks"], mask_id)
    logger.info(f"Tentative d'accès au masque : {mask_path}")
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Masque non trouvé")

    # Charger le masque
    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    # Vérifier que toutes le valeurs du masque sont dans la plage attendue
    if np.any((mask_array < 0) | (mask_array >= len(LABELS))):
        logger.error(f"Le masque contient des valeurs non valides : {np.unique(mask_array)}")
        raise HTTPException(status_code=400, detail="Le masque contient des valeurs non valides")

    # Mapper les valeurs du masque aux nouvelles catégories
    mapped_mask_array = np.array([label.categoryId for label in LABELS])[mask_array]

    # Convertir le masque en image colorée
    mask_image = Image.fromarray(COLOR_PALETTE[mapped_mask_array].astype(np.uint8))

    # Sauvegarder l'image du masque dans un buffer
    buffer = BytesIO()
    mask_image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")

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
