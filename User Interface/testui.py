import pytest
from unittest.mock import patch, Mock
from PIL import Image
import os
import requests
import io
from app import get_image_list, display_image, get_mask_name, display_mask, make_prediction  # Importez depuis app.py

# Définition des chemins
dirs = {
    "images": "../Images/Photos",
    "masks": "../Images/Mask",
    "model": "../Model/efficientnet_fpn_model_best_iou_diceloss.keras"
}

# Créer une image valide pour les tests
def create_valid_image():
    image = Image.new('RGB', (100, 100), color='red')
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    byte_arr.seek(0)
    return byte_arr.read()

def test_get_image_list_success(mocker):
    # Simuler une réponse réussie de l'API
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"images": ["image1.png", "image2.png"]}
    mock_response.raise_for_status.return_value = None

    mocker.patch('app.requests.get', return_value=mock_response)

    # Appeler la fonction et vérifier le résultat
    images = get_image_list()
    assert images == ["image1.png", "image2.png"]

def test_get_image_list_failure(mocker):
    # Simuler une exception de requête
    mocker.patch('app.requests.get', side_effect=requests.exceptions.RequestException("Erreur de connexion"))

    # Appeler la fonction et vérifier le résultat
    images = get_image_list()
    assert images == []

def test_display_image_success(mocker):
    # Simuler une réponse réussie de l'API avec une image valide
    mock_response = mocker.Mock()
    mock_response.content = create_valid_image()
    mock_response.raise_for_status.return_value = None

    mocker.patch('app.requests.get', return_value=mock_response)

    # Appeler la fonction et vérifier le résultat
    image = display_image("image1.png")
    assert image is not None

def test_display_image_failure(mocker):
    # Simuler une exception de requête
    mocker.patch('app.requests.get', side_effect=requests.exceptions.RequestException("Erreur de connexion"))

    # Appeler la fonction et vérifier le résultat
    image = display_image("image1.png")
    assert image is None

def test_get_mask_name():
    # Vérifier que le nom du masque est correctement généré
    mask_name = get_mask_name("image1_leftImg8bit.png")
    assert mask_name == "image1_gtFine_labelIds.png"


def test_display_mask_failure():
    # Vérifier que le masque non trouvé retourne None
    mask_name = "non_existent_mask.png"
    mask = display_mask(mask_name)
    assert mask is None

def test_make_prediction_success(mocker):
    # Simuler une réponse réussie de l'API avec une image valide
    mock_response = mocker.Mock()
    mock_response.content = create_valid_image()
    mock_response.raise_for_status.return_value = None

    mocker.patch('app.requests.post', return_value=mock_response)

    # Créer une image fictive pour le test
    image = Image.new('RGB', (100, 100))

    # Appeler la fonction et vérifier le résultat
    prediction_image = make_prediction(image)
    assert prediction_image is not None

def test_make_prediction_failure(mocker):
    # Simuler une exception de requête
    mocker.patch('app.requests.post', side_effect=requests.exceptions.RequestException("Erreur de connexion"))

    # Créer une image fictive pour le test
    image = Image.new('RGB', (100, 100))

    # Appeler la fonction et vérifier le résultat
    prediction_image = make_prediction(image)
    assert prediction_image is None
