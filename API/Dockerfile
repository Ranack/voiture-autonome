# Utiliser une image de base légère avec Python 3.9
FROM python:3.9-slim

# Mettre à jour les paquets et installer les dépendances système nécessaires
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

# Copier le code de l'application dans le conteneur
COPY API/ .

# Copier les répertoires des images et des masques dans le conteneur
COPY Images/Photos /app/Images/Photos
COPY Images/Mask /app/Images/Mask

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application va tourner
EXPOSE 8000

# Commande pour lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
