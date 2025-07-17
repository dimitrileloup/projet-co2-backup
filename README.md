# Présentation du projet
Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution. 
Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple).

Ce projet s'inscrit dans notre cursus Data Scientest - Novembre 2024.

Données : https://www.kaggle.com/datasets/dimitrileloup/vehicules-fr-2022-2023

# Membres
Younès ABIDAT
Victor BRUNET
Christian GIBOUDEAU
Dimitri LELOUP

Mentor : Antoine F

# Technologies
Python, Pandas, Plotly, Scikit-learn, Streamlit

# Organisation des répertoires

# Streamlit
Streamlit permet de créer des applications web interactives.
## Installation en local de Streamlit
```pip install streamlit```
## Création et activation d'un environnement "co2_env"
Lancer Anaconda Prompt

Créer, si nécessaire l’environnement co2

```conda create -n co2_env python=3.10```

Activer l’environnement 

```conda activate co2_env```

Se positionner dans le répertoire streamlit

## Installer les "requirements" 

```conda install scikit-learn=1.5.1```

```pip install streamlit```

```pip install pandas```

```pip install numpy```

```pip install joblib```

```pip install matplotlib```

```pip install xgboost```

```pip install plotly```

```pip install seaborn```

```pip install kagglehub```

```pip install streamlit-javascript```

## Lancer l'application

```streamlit run app.py```

## Tips
Lister les environnements

```conda env list```

En cas de difficulté à quitter le streamlit

```tasklist | findstr streamlit```

puis 

```taskkill /PID 12345 /F```
