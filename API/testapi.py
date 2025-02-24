import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image
import numpy as np
import os
import sys

# Ajouter le répertoire 'API' au PYTHONPATH pour les tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer le code de l'application avec un import absolu
from API.main import app, get_image_list, install_package

# Créer un client de test pour l'application FastAPI
client = TestClient(app)

## Définition des chemins
DIRS = {
    "images": "/app/Images/Photos",  # Mise à jour du chemin pour correspondre à votre conteneur Docker
    "masks": "/app/Images/Mask",     # Mise à jour du chemin pour correspondre à votre conteneur Docker
    "model": "../API/Model/efficientnet_fpn_model_best_iou_diceloss.keras"  # Chemin absolu pour le modèle
}

def test_health_check():
    """
    Teste l'endpoint /health pour vérifier si l'API est opérationnelle.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}

@patch('API.main.os.walk')
@patch('API.main.tf.keras.models.load_model')
def test_list_images(mock_load_model, mock_walk):
    """
    Teste l'endpoint /images pour vérifier qu'il retourne la liste correcte des images.
    Utilise un mock pour simuler le contenu du répertoire des images et le chargement du modèle.
    """
    # Simuler la structure du répertoire avec des sous-dossiers
    mock_walk.return_value = [
        (dirs["images"], ('subdir',), ('image1.png', 'image2.jpg')),
        (os.path.join(dirs["images"], 'subdir'), (), ('image3.png',))
    ]

    # Mock du chargement du modèle
    mock_load_model.return_value = MagicMock()

    response = client.get("/images")
    assert response.status_code == 200
    assert response.json() == {
        "images": [
            'image1.png',
            'image2.jpg',
            os.path.join('subdir', 'image3.png')
        ]
    }

@patch('API.main.os.path.exists')
def test_get_image_not_found(mock_exists):
    """
    Teste l'endpoint /images/{image_id} pour vérifier qu'il retourne une erreur 404 si l'image n'existe pas.
    Utilise un mock pour simuler l'absence du fichier.
    """
    mock_exists.return_value = False
    response = client.get("/images/image1.png")
    assert response.status_code == 404

@patch('API.main.os.path.exists')
def test_get_mask_not_found(mock_exists):
    """
    Teste l'endpoint /masks/{mask_id} pour vérifier qu'il retourne une erreur 404 si le masque n'existe pas.
    Utilise un mock pour simuler l'absence du fichier.
    """
    mock_exists.return_value = False
    response = client.get("/masks/mask1.png")
    assert response.status_code == 404

@patch('subprocess.check_call')
def test_install_package(mock_check_call):
    """
    Teste la fonction install_package pour vérifier qu'elle appelle correctement subprocess.check_call
    pour installer un package avec pip.
    """
    install_package("some-package")
    mock_check_call.assert_called_once_with([sys.executable, "-m", "pip", "install", "some-package"])
