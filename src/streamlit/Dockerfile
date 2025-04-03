# Utilise une image de base Python slim
FROM python:3.11-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Créer un dossier pour l'app
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du projet
COPY . .

# Exposer le port Streamlit
EXPOSE 8501

# Lancer l'application
CMD ["sh", "-c", "echo '\nStreamlit lancé !\nAccès depuis votre machine : http://localhost:8501\n' && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]