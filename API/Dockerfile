# Utiliser une image officielle Python comme image de base
FROM python:3.9-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY API/ .

# Copier le modèle dans le conteneur
COPY Model/efficientnet_fpn_model_best_iou_diceloss.keras /app/Model/

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn tensorflow==2.18.0 segmentation-models==1.0.1 python-multipart

# Si vous avez un fichier requirements.txt, vous pouvez l'utiliser ici :
# RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par l'application
EXPOSE 8000

# Commande pour démarrer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
