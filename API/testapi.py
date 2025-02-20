import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image
import numpy as np
import os
import sys

# Importer le code de l'application
from main import app, get_image_list, install_package

# Créer un client de test pour l'application FastAPI
client = TestClient(app)

# Définition des chemins
dirs = {
    "images": "../Images/Photos",
    "masks": "../Images/Mask",
    "model": "../Model/efficientnet_fpn_model_best_iou_diceloss.keras"
}

def test_health_check():
    """
    Teste l'endpoint /health pour vérifier si l'API est opérationnelle.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}

@patch('main.os.listdir')
def test_list_images(mock_listdir):
    """
    Teste l'endpoint /images pour vérifier qu'il retourne la liste correcte des images.
    Utilise un mock pour simuler le contenu du répertoire des images.
    """
    mock_listdir.return_value = ['image1.png', 'image2.jpg']
    response = client.get("/images")
    assert response.status_code == 200
    assert response.json() == {"images": ['image1.png', 'image2.jpg']}

@patch('main.os.path.exists')
def test_get_image_not_found(mock_exists):
    """
    Teste l'endpoint /images/{image_id} pour vérifier qu'il retourne une erreur 404 si l'image n'existe pas.
    Utilise un mock pour simuler l'absence du fichier.
    """
    mock_exists.return_value = False
    response = client.get("/images/image1.png")
    assert response.status_code == 404

@patch('main.os.path.exists')
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
