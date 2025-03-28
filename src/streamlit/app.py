import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import kagglehub
import os
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
st.set_page_config(layout="wide")

from streamlit_javascript import st_javascript

# On récupère l'URL du navigateur 
url = st_javascript("window.location.href")
environnement = ''
if url:
    if "localhost" in url or "127.0.0.1" in url:
        environnement = "local"
    else:
        environnement = "cloud"
    
    st.success(f"Environnement détecté : {environnement}")
    st.write(f"URL : {url}")
else:
    st.warning("En attente du navigateur...")

if environnement == "local":
    # Chemins des fichiers en local
    csv_path_dataset_nettoye = "datasets/datas_nettoyees_model_FR.csv"
    # Nous passons par Kaggle car le dataset ne peut être envoyé sur Github : il est trop volumineux. Cela est plus rapide en local aussi.
    path = kagglehub.dataset_download("dimitrileloup/vehicules-fr-2022-2023")
    csv_path_dataset_original = f"{path}/datas_FR_2022_2023.csv"
    dossier_documents = "documents/"
else:
    # chemin des fichiers pour le déploiement sur Streamlit
    #csv_path_dataset_nettoye = "https://raw.githubusercontent.com/dimitrileloup/projet-co2-backup/refs/heads/main/notebooks/datasets/Dataset_final/datas_nettoyees_model_FR.csv"
    #csv_path_dataset_original = "https://huggingface.co/datasets/dleloup/vehicules-co2/resolve/main/datas_FR_2022_2023.csv"
    csv_path_dataset_nettoye = "https://huggingface.co/datasets/dleloup/vehicules-co2/resolve/main/datasets/datas_nettoyees_model_FR.csv"
    csv_path_dataset_original = "https://huggingface.co/datasets/dleloup/vehicules-co2/resolve/main/datasets/datas_FR_2022_2023.csv"
    dossier_documents = "https://huggingface.co/datasets/dleloup/vehicules-co2/resolve/main/documents/"


def load_and_format_csv(file_path):
    # Charger le CSV
    df = pd.read_csv(file_path)

    # Fonction de formatage
    def format_value(x):
        if isinstance(x, float):  # Float : 5 décimales
            return f"{x:,.5f}".replace(",", "")
        elif isinstance(x, int):  # Int : formaté avec des espaces
            return f"{x:,}".replace(",", "")
        else:
            return x

    # Appliquer le formatage colonne par colonne
    for col in df.select_dtypes(include=["float", "int"]):
        df[col] = df[col].map(format_value)

    return df


def detecter_outliers_plotly_var(serie, seuil=1.5):
    """
    Détecte les outliers pour une seule variable numérique en utilisant la méthode IQR.
    Affiche également un Boxplot interactif.

    :param serie: Série Pandas contenant les valeurs de la variable.
    :param seuil: Seuil du coefficient IQR (par défaut 1.5).
    :return: DataFrame contenant le nombre d'outliers et le pourcentage.
    """
    if not isinstance(serie, pd.Series):
        raise ValueError("Veuillez fournir une variable sous forme de pd.Series")

    Q1 = serie.quantile(0.25)  # Premier quartile
    Q3 = serie.quantile(0.75)  # Troisième quartile
    IQR = Q3 - Q1  # Intervalle interquartile

    # Détection des valeurs aberrantes
    lower_bound = Q1 - seuil * IQR
    upper_bound = Q3 + seuil * IQR
    outliers = serie[(serie < lower_bound) | (serie > upper_bound)]

    # Résultats
    nb_outliers = outliers.shape[0]
    perc_outliers = (nb_outliers / serie.shape[0]) * 100

    # Création du Boxplot avec Plotly
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=serie,
        name=serie.name if serie.name else "Variable",
        marker_color='blue',
        boxpoints='outliers' 
    ))

    # Mise en page
    fig.update_layout(
        title=f"Box Plot de {serie.name if serie.name else 'Variable'}",
        xaxis_title="Valeur",
        yaxis_title="Distribution",
        template="plotly_white",
        showlegend=False
    )

    # Résultats sous forme de DataFrame
    df_outliers = pd.DataFrame({
        "Nombre Outliers": [nb_outliers],
        "Pourcentage": [round(perc_outliers, 2)]
    }, index=[serie.name if serie.name else "Variable"])

    return df_outliers, fig

st.markdown("""
        <style>
        /* Modifier tous les expanders */
        [data-testid="stExpander"] {
            
        }
        h2 {
            font-size:25px;  
        }
        h3 {
            font-size:20px;    
        }
        /* Modifier uniquement le titre de l'expander */
        [data-testid="stExpander"] {
            border: 0.5px solid #e6e6e6 !important; /* Bordure bleue */
            border-radius: 10px !important;
        }
        [data-testid="stExpander"] em {
            font-size: 1.35rem !important;
            font-weight: 600 !important;
            color: #31333F !important; 
            font-style: normal !important; 

        }
        </style>
        """, unsafe_allow_html=True)

st.markdown("""
            <style>
            .stRadio p{
                font-size: 15px !important;
            }
            section[data-testid="stSidebarUserContent"] {
                font-size: 15px !important;
            }
            .stVerticalBlock li {
                 font-size: 15px !important;
            }
            </style>
        """, unsafe_allow_html=True)

st.title("Emissions de CO2 par les véhicules")

# Sidebar
st.sidebar.title("Menu")

section = st.sidebar.radio(
    " ➡️ Aller à :", 
    [
        "🏠 Introduction et objectif", 
        "📂 Choix, chargement et aperçu du dataset orginal", 
        "🔍 Exploration des données", 
        "⚙️ Pré-processing", 
        "📊 Analyse des variables",
        "🌿 Analyse de la variable cible CO2", 
        "🚨 Analyse des outliers", 
        "🤖 Modélisation",
        "🚀 Démo",
        "📝 Conclusion"
    ]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### 👥 Membres du projet")
st.sidebar.markdown("""
- Younès ABIDAT
- Victor BRUNET
- Christian GIBOUDEAU 
- Dimitri LELOUP
""")
st.sidebar.markdown(""" 
- **Mentor** : Antoine FRADIN
""")
st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎓 Promotion Continue Data Scientest Novembre 2024")

try:
    df_original = pd.read_csv(csv_path_dataset_original, nrows=1000)
    df_original.columns = df_original.columns.str.strip()

    df_nettoye = pd.read_csv(csv_path_dataset_nettoye)
    df_nettoye.columns = df_nettoye.columns.str.strip()
    
    if section == "🏠 Introduction et objectif":
        st.header("🏠 Introduction et objectif")
        st.write("Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution. "
        "Prédire à l’avance cette pollution permet de prévenir dans le cas de l’apparition de nouveaux types de véhicules (nouvelles séries de voitures par exemple).")

    elif section == "📂 Choix, chargement et aperçu du dataset orginal":
        st.header("📂 Choix, chargement et aperçu du dataset orginal")        

        st.subheader("Choix du jeu de données")
        st.write("Nous avons étudié et recherché plusieurs jeux de données pour notre projet.")
        st.markdown('Dans un premier temps, nous avions une préférence pour les <a href="https://www.data.gouv.fr/" target="_blank">jeux de données du Gouvernement.</a>', unsafe_allow_html=True)
        st.write("En effet, leurs jeux de données paraissaient mieux structurés et plus facilement compréhensibles.")
        st.write("Mais en les analysant de manière plus approfondie, nous nous sommes rendus compte de plusieurs choses :")
        st.markdown("""
                - Les marques de voitures étaient mal représentées : Mercedes et Volkswagen comptabilisaient 90% des modèles
                - Les jeux de données commençaient à dater (le plus récent était de 2014)
                """)
        st.markdown('C\'est pour ces raisons que nous sommes partis sur les jeux de données du site <a href="https://www.eea.europa.eu/data-and-maps/data/co2-cars-emission-20" target="_blank">europa.eu</a> : ', unsafe_allow_html=True)
        st.markdown("""
                - Les jeux étaient beaucoup plus denses (plusieurs centaines de milliers de lignes)
                - Les marques étaient mieux représentées
                """)
        
        st.subheader("Chargement et aperçu des données")
        #st.write("Pour des raisons de performance, seules les 1000 premières ont été chargées. Pour informations le dataset d'origine compte 3 528 480 lignes.")
        
        code_snippet = """
                        df = pd.read_csv("datas_FR_2022_2023.csv")
                        df.head(10)"""
        st.code(code_snippet, language="python")   
        for col in df_original.select_dtypes(include=["int", "float"]):
            df_original[col] = df_original[col].map(lambda x: f"{x:,}".replace(",", " "))

        st.dataframe(df_original.head(10))        

        st.subheader("Signification des colonnes")
        code_snippet = """
        var = pd.read_excel("Table-definition.xlsx")
        var.head(40)"""
        st.code(code_snippet, language="python")
        
        df_analyse = pd.read_csv(f"{dossier_documents}description_colonnes.csv")
        df_analyse
    
    elif section == "🔍 Exploration des données":
        st.header("🔍 Exploration des données")
        st.subheader("Analyse rapide des colonnes")
        show_code = st.toggle("Afficher le code de la fonction utilisée")

        if show_code:
            st.code("""
                    def analyse_columns(df):
                        analysis = []

                        for col in df.columns:
                            col_name = col
                            col_type = df[col].dtype
                            unique_count = df[col].nunique()  # Nombre de valeurs uniques
                            unique_values = df[col].dropna().unique()[:5].tolist()  # Exemples (max 5)
                            unique_values = " | ".join(map(str, unique_values))  # Séparateur : " | "

                            analysis.append({
                                'Nom de la colonne': col_name,
                                'Type de la colonne': col_type,
                                'Nombre de valeurs uniques': unique_count,
                                'Exemples de valeurs': unique_values
                            })

                        return pd.DataFrame(analysis)
                        """, language="python")
        df_analyse = load_and_format_csv(f"{dossier_documents}analyse_rapide_colonnes.csv")
        #df_analyse = pd.read_csv("documents/analyse_rapide_colonnes.csv")
        df_analyse     

        st.subheader("Description rapide des variables numériques")
        code_snippet = """df.describe(include='number').T"""
        st.code(code_snippet, language="python")
        df_analyse = load_and_format_csv(f"{dossier_documents}describe_numerical.csv")
        #df_analyse = pd.read_csv("documents/describe_numerical.csv")
        df_analyse

        st.subheader("Description rapide des variables catégorielles")
        code_snippet = """df.describe(include='object').T"""
        st.code(code_snippet, language="python")
        df_analyse = load_and_format_csv(f"{dossier_documents}describe_object.csv")
        #df_analyse = pd.read_csv("documents/describe_object.csv")
        df_analyse        
    
    elif section == "⚙️ Pré-processing":
        st.header("⚙️ Pré-processing")
        with st.expander("🔍 *Analyse des valeurs manquantes*"):
            show_code = st.toggle("Afficher le code de la fonction créée")

            if show_code:
                st.code("""
                        def display_missing_values(df):
                            missing_values = df.isnull().sum()
                            missing_ratio = (missing_values / len(df)) * 100

                            missing_values_df = pd.DataFrame({
                                'Colonne': missing_values.index,
                                'Valeurs manquantes (%)': missing_ratio.values,
                                'Nombre de valeurs manquantes': missing_values.values,
                                'Type': df.dtypes.values
                            })

                            missing_values_df = missing_values_df[missing_values_df['Nombre de valeurs manquantes'] > 0]
                            missing_values_df = missing_values_df.sort_values(by='Valeurs manquantes (%)', ascending=False).reset_index(drop=True)

                            return missing_values_df
                            """, language="python")
            df_analyse = load_and_format_csv(f"{dossier_documents}analyse_valeurs_manquantes.csv")
            #df_analyse = pd.read_csv("documents/analyse_valeurs_manquantes.csv")
            df_analyse
        
        with st.expander("🗑️ *Suppression de colonnes*"):
            st.subheader("Suppression des colonnes avec un taux de valeurs manquantes supérieur à 70%")
            st.write("Nous pouvons dès à présent ces colonnes qui ont un taux de valeurs manquantes supérieur à 70% :")
            st.markdown("""
                    - At2 (mm)
                    - W (mm)
                    - MMS
                    - Vf
                    - De
                    - Ernedc (g/km)
                    - At1 (mm)
                    - Enedc (g/km)
                    - RLFI
                    - z (Wh/km)
                    - Electric range (km)
                    """)
            code_snippet = """df = df.drop(columns=['At2 (mm)', 'W (mm)', 'MMS', 'Vf', 'De', 'Ernedc (g/km)', 'At1 (mm)', 'Enedc (g/km)', 'RLFI', 'z (Wh/km)', 'Electric range (km)'], axis=1)"""
            st.code(code_snippet, language="python")
        
            st.subheader("Suppression des colonnes non pertinentes")
            st.write("Certaines colonnes n'ont pas d'intérêt à être gardées :")
            st.markdown("""
                    - IT
                    - Erwltp (g/km) (dépreciée)
                    - ID : identifiant du véhicule
                    - Country : notre dataset est une extraction des véhicules de France
                    - VFN : n'a pas de norme universelle et comporte trop de valeurs
                    - Tan : trop de valeurs et sans intérêt pour notre projet
                    - T : trop de valeurs et sans intérêt pour notre projet
                    - Va : trop de valeurs et sans intérêt pour notre projet
                    - Ve : trop de valeurs et sans intérêt pour notre projet
                    - Status : n'a qu'une seule valeur et ne varie pas
                    - Year : 1 seule valeur
                    - Date of registration : sans intérêt pour notre projet
                    - Fm : redondant avec Ft
                    - Cr : nous avons 2 catégories (M1, M1G). M1G est une sous-catégorie de M1 réservée aux véhicules tout-terrain. Nous pouvons conclure que tous les véhicules sont de catégorie M1
                    - Ct : idem que Cr
                    - ech : sans intérêt pour notre projet
                    - Mp : redondant, se retrouve dans une autre colonne
                    - Man : redondant avec Mk
                    - r : n'a qu'une seule valeur
                    - Mh : redondant avec Mk
                    """)
            code_snippet = """df = df.drop(columns=['IT', 'Erwltp (g/km)', 'ID', 'Country', 'VFN', 'Tan', 'T', 'Va', 'Ve', 'Status', 'year', 'Date of registration', 'Fm', 'Cr', 'Ct', 'ech', 'Mp', 'Man', 'r', 'Mh'], axis=1)"""
            st.code(code_snippet, language="python")
        
        with st.expander("🔗 *Vérification de la corrélation entre Masse à vide et Masse totale*"):       
            st.subheader("Vérification de la corrélation entre Masse à vide et Masse totale")
            corr_matrix = df_original[['Mt', 'm (kg)']].corr().round(2)
            fig_corr = px.imshow(
                                corr_matrix, 
                                text_auto=True, 
                                title="Matrice de corrélation des variables numériques", 
                                color_continuous_scale='blues')
            fig_corr.update_layout(
                                    xaxis_tickangle=-45,
                                    width=600,
                                    height=600)
            st.plotly_chart(fig_corr)

            st.subheader("Suppression de la colonne Mt")
            st.write("On se rend compte qu'il y a une **forte corrélataion** entre ces 2 variables, qui pourrait entrainer une **colinéarité**. Nous prenons la décision de ne garder que la masse à vide (m (kg)).")
            code_snippet = """df = df.drop('Mt', axis=1)"""
            st.code(code_snippet, language="python")

        with st.expander("✏️ *Renommage de colonnes*"):      
            st.write("Pour plus de compréhension, nous allons renommer les colonnes :")
            st.markdown("""
                        - Mk : Marque
                        - Cn : Modele
                        - Ewltp (g/km) : Co2
                        - Ft : Carburant
                        - ec (cm3) : Cylindree moteur
                        - ep (KW) : Puissance moteur
                        - Fuel consumption : Consommation carburant""")
            code_snippet = """
                renommage = {
                    'Mk': 'Marque',
                    'Cn': 'Modèle',
                    'm (kg)' : 'Masse à vide',
                    'Ewltp (g/km)': 'CO2',
                    'Ft': 'Carburant',
                    'ec (cm3)': 'Cylindrée moteur',
                    'ep (KW)': 'Puissance moteur',
                    'Fuel consumption': 'Consommation carburant'
                }

                # Application du renommage
                df.rename(columns=renommage, inplace=True)
                """
            st.code(code_snippet, language="python")

        with st.expander("🤔 *Faut-il garder les véhicules électriques et hydrogènes ?*"):    
            st.write("Notre objectif est de prédire les émissions directes de CO2. ")
            st.write("Bien que les véhicules électriques et hydrogènes produisent du CO2 à l'étape de leur construction, il ne s'agit pas de CO2 émis en fonctionnement.")
            st.write("Garder les véhicules électriques et hydrogènes risque de biaiser notre modèle. Nous avons donc décide d'**exclure** les véhicules électriques et hydogènes de notre dataset")

            code_snippet = """
                            # Nous excluons les véhicules électriques et hydrogènes
                            df = df[(df["Carburant"] != "electric") & (df["Carburant"] != "hydrogen")]"""
            st.code(code_snippet, language="python")

        with st.expander("📑 *Gestion des doublons*"):    
            code_snippet = """
                            df.duplicated().sum()
                            df.drop_duplicates(inplace=True)
                            df = df.reset_index(drop=True)"""
            st.code(code_snippet, language="python")

        with st.expander("🛠️ *Traitement des valeurs manquantes*"):       
            st.markdown("##### Valeurs manquantes")
            df_vm = pd.read_csv(f"{dossier_documents}valeurs_manquantes.csv")
            df_vm
            
            st.markdown("##### Gestion des valeurs manquantes de la colonne Consommation carburant")
            code_snippet = """
                            df_fc_na = df[df['Consommation carburant'].isna()]
                            df_fc_na"""
            st.code(code_snippet, language="python")
            #df_vm = pd.read_csv("documents/valeurs_manquantes2.csv")
            df_vm = load_and_format_csv(f"{dossier_documents}valeurs_manquantes2.csv")
            df_vm

            code_snippet = """
                        # regardons si d'autres modeles RANGE ROVER EVOQUE sont renseignés
                        df_jag_evoque = df[(df['Modèle'] == 'RANGE ROVER EVOQUE') & (df['Carburant'] == 'diesel') & (~df['Consommation carburant'].isna())]
                        df_jag_evoque.head()
                        """
            st.code(code_snippet, language="python")
            #df_vm = pd.read_csv("documents/valeurs_manquantes3.csv")
            df_vm = load_and_format_csv(f"{dossier_documents}valeurs_manquantes3.csv")
            df_vm

            code_snippet = """
                        fc_mean_jag_evoque = df_jag_evoque['Consommation carburant'].mean().round(1)
                        fc_mean_jag_evoque
                        6.5"""
            st.code(code_snippet, language="python")
            
            code_snippet = """
                        df.loc[(df["Modèle"] == "RANGE ROVER EVOQUE") & (df['Consommation carburant'].isna()), "Consommation carburant"] = fc_mean_jag_evoque
            """
            st.code(code_snippet, language="python")

            st.markdown("##### Gestion des valeurs manquantes de la colonne Masse à vide")
            df_vm = pd.read_csv(f"{dossier_documents}valeurs_manquantes4.csv")
            df_vm

            code_snippet = """
                            df_mkg_na = df[df['Masse à vide'].isna()]
                            df_mkg_na"""
            st.code(code_snippet, language="python")
            #df_vm = pd.read_csv("documents/valeurs_manquantes5.csv")
            df_vm = load_and_format_csv(f"{dossier_documents}valeurs_manquantes5.csv")
            df_vm

            st.write("Recherchons dans le dataset si nous avons des modèles équivalents dont les variables m (kg) et Mt ne sont **pas nulles**")
            code_snippet = """
                        df_jag_lr = df[(df['Modèle'] == 'RANGE ROVER EVOQUE') & (df['Carburant'] == 'diesel') & (df['Puissance moteur'] == 120) & (~df['Masse à vide'].isna())]
                        df_jag_lr
                        """
            st.code(code_snippet, language="python")
            df_vm = load_and_format_csv(f"{dossier_documents}valeurs_manquantes6.csv")
            #df_vm = pd.read_csv("documents/valeurs_manquantes6.csv")
            df_vm

            code_snippet = """
                        fc_mean_masse_jag_evoque = df_jag_evoque['Masse à vide'].mean().round(1)
                        fc_mean_masse_jag_evoque
                        1955.3"""
            st.code(code_snippet, language="python")
            
            code_snippet = """
                        # pour les mêmes modèles, la variable Masse à vide est égale à 1967
                        df['Masse à vide'] = df['Masse à vide'].fillna(fc_mean_masse_jag_evoque)"""
            st.code(code_snippet, language="python")
            
            code_snippet = """
                        # Vérification
                        df.loc[23942].to_frame().T
                        """
            st.code(code_snippet, language="python")
            df_vm = load_and_format_csv(f"{dossier_documents}valeurs_manquantes7.csv")
            #df_vm = pd.read_csv("documents/valeurs_manquantes7.csv")
            df_vm
        
        with st.expander("📂 *Regroupement de catégories*"):    
            st.write("Certaines valeurs de la variable **Carburant** peuvent être regroupées :")
            st.markdown("""
                        - diesel/electric & petrol/electric sont des véhicules hybride
                        - lpg & ng sont des énergies au gaz
                        - petrol sera renommé en essence pour plus de compréhension.""")
            code_snippet = """
                            replace_ft = {
                                        'petrol' : 'essence',
                                        'diesel/electric' : 'hybride',
                                        'petrol/electric' : 'hybride',
                                        'lpg' : 'gaz',
                                        'ng' : 'gaz'}
                            df['Carburant'] = df['Carburant'].replace(replace_ft)
                            """
            st.code(code_snippet, language="python")

            st.write("Certaines valeurs la variable **Marque** peuvent être regroupées, comme par exemple :")
            st.markdown("""
                        - MC LAREN, MCLAREN
                        - MERCEDES AMG, MERCEDES BENZ, MERCEDES-BENZ
                        - MITSUBISHI, MITSUBISHI MOTORS CORPORATION, MITSUBISHI MOTORS THAILAND""")
            code_snippet = """
                            replace_mk = {'MC LAREN' : 'MCLAREN',
                                        'MERCEDES AMG' : 'MERCEDES BENZ',
                                        'MERCEDES-BENZ' : 'MERCEDES BENZ',
                                        'MITSUBISHI MOTORS CORPORATION' : 'MITSUBISHI',
                                        'MITSUBISHI MOTORS THAILAND' : 'MITSUBISHI',
                                        'MITSUBISHI MOTORS (THAILAND)' : 'MITSUBISHI',
                                        'FORD-CNG-TECHNIK' : 'FORD',
                                        'ROLLS ROYCE' : 'ROLLS-ROYCE'}
                            df['Marque'] = df['Marque'].replace(replace_mk)
                            """
            st.code(code_snippet, language="python")
        with st.expander("💡 *Création d'indicateurs*"):    
            st.write("Certains indicateurs peuvent permettre d'analyser les émissions de CO2 :")
            st.markdown("##### Indicateur de Charge Spécifique du Moteur (ICSM)")
            st.code("""ICSM = Puissance (kW) / Masse du véhicule (kg)""", language="python")
            st.write("**Interprétation** : ")
            st.markdown("""
                        - Faible ICSM → Voiture puissante et légère (moins d’effort, moins de CO₂).
                        - Élevé ICSM → Voiture sous-motorisée (forte sollicitation, plus de CO₂).
                        """)

            st.code("""df['ICSM'] = df['Puissance moteur'] / df['Masse à vide']""", language="python")

            st.markdown("##### Indicateur de Consommation Énergétique (ICE)")
            st.code("""ICE = Puissance (kW) / Cylindrée (cm³)""", language="python")
            st.write("**Interprétation** : ")
            st.markdown("""
                        - Faible ICE → Moteur optimisé (ex. turbo downsizing).
                        - Élevé ICE → Moteur gourmand et peu efficient.
                        """)

            st.code("""df['ICE'] = df['Puissance moteur'] / df['Cylindrée moteur']""", language="python")

            st.markdown("##### Indicateur de Densité Energétique du Carburant (IDEC)")
            st.code("""IDEC = Cylindrée (cm³) / Masse du véhicule (kg)""", language="python")
            st.write("**Interprétation** : ")
            st.markdown("""
                        - Faible IDEC → Moteur bien dimensionné (moins d’effort, moins de CO₂).
                        - Élevé IDEC → Moteur sous-dimensionné (forte sollicitation, plus de CO₂).
                        """)

            st.code("""df['IDEC'] = df['Cylindrée moteur'] / df['Masse à vide']""", language="python")

    elif section == "📊 Analyse des variables":
        st.header("📊 Analyse des variables")

        st.subheader("Analyse des variables catégorielles")
        categorical_columns = df_nettoye.select_dtypes(include='object').columns.tolist()

        show_code = st.toggle("Afficher la description rapide des variables catégorielles")
        if show_code:
            st.markdown("#### Description rapide")
            code_snippet = """
            categorical_columns = df_nettoye.select_dtypes(include='object').columns.tolist()
            df_nettoye[categorical_columns].describe().T"""
            st.code(code_snippet, language="python")
            st.write(df_nettoye[categorical_columns].describe().T)

        st.markdown("#### Visualisations")
        if categorical_columns:
            selected_variable = st.selectbox("Choisissez une variable catégorielle à analyser:", categorical_columns)
            top_n = st.slider("Nombre de modalités à afficher:", min_value=5, max_value=50, value=20)
            top_categories = df_nettoye[selected_variable].value_counts().head(top_n).reset_index()
            top_categories.columns = [selected_variable, 'Count']
            fig = px.bar(
                top_categories, 
                x=selected_variable, 
                y='Count', 
                title=f"Top {top_n} modalités de {selected_variable}",
                labels={'Count': 'Nombre'} 
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig)
        else:
            st.warning("Aucune variable catégorielle trouvée dans les données.")
        
        st.subheader("Analyse des variables numériques")
        numerical_columns = df_nettoye.select_dtypes(include='number').columns.tolist()
        show_code = st.toggle("Afficher la description rapide des variables numériques")
        if show_code:
            st.markdown("#### Description rapide")
            code_snippet = """
            numerical_columns = df_nettoye.select_dtypes(include='number').columns.tolist()
            df_nettoye[numerical_columns].describe().T"""
            st.code(code_snippet, language="python")
            st.write(df_nettoye[numerical_columns].describe().T)

        st.markdown("#### Visualisations")

        selected_num_var = st.selectbox("Choisissez une variable numérique pour l'histogramme:", numerical_columns)
        bins = st.slider("Nombre de bins pour l'histogramme :", min_value=5, max_value=100, value=30, step=5)

        # Création de l'histogramme avec transparence et bordure
        fig = px.histogram(
            df_nettoye, 
            x=selected_num_var, 
            title=f"Distribution de {selected_num_var}", 
            nbins=bins,
            labels={'Count': 'Nombre'} )  

        # Ajout d'une bordure noire
        fig.update_traces(marker=dict(line=dict(width=1, color='#E0E0E0')))

        # Affichage du graphique
        st.plotly_chart(fig)

    elif section == "📈 Analyse des variables numériques":
        st.header("📈 Analyse des variables numériques")
        numerical_columns = df_nettoye.select_dtypes(include='number').columns.tolist()
        show_code = st.toggle("Afficher la description rapide")
        if show_code:
            st.subheader("Description rapide")
            code_snippet = """
            numerical_columns = df_nettoye.select_dtypes(include='number').columns.tolist()
            df_nettoye[numerical_columns].describe().T"""
            st.code(code_snippet, language="python")
            st.write(df_nettoye[numerical_columns].describe().T)

        st.subheader("Visualisations")

        selected_num_var = st.selectbox("Choisissez une variable numérique pour l'histogramme:", numerical_columns)
        bins = st.slider("Nombre de bins pour l'histogramme :", min_value=5, max_value=100, value=30, step=5)

        # Création de l'histogramme avec transparence et bordure
        fig = px.histogram(
            df_nettoye, 
            x=selected_num_var, 
            title=f"Distribution de {selected_num_var}", 
            nbins=bins,
            labels={'Count': 'Nombre'} )  

        # Ajout d'une bordure noire
        fig.update_traces(marker=dict(line=dict(width=1, color='#E0E0E0')))

        # Affichage du graphique
        st.plotly_chart(fig)

    elif section == "🌿 Analyse de la variable cible CO2":
        st.header("🌿 Analyse de la variable cible CO2")
        st.subheader("Analyse du CO2 en fonction des autres variables")
        target_variable = "CO2"
        if target_variable in df_nettoye.columns:
            co2_mean = round(df_nettoye[target_variable].mean(), 2)
            selected_feature = st.selectbox("**Choisissez une variable :**", [col for col in df_nettoye.columns if col != target_variable])
            if selected_feature not in ["Marque", "Modèle", "Carburant"]:
                fig = px.scatter(
                    df_nettoye, 
                    x=selected_feature, 
                    y=target_variable, 
                    color="Carburant", 
                    size="CO2",
                    title=f"CO2 en fonction de {selected_feature} (coloré par Carburant)",
                    hover_data=df_nettoye.columns
                )
                fig.add_hline(
                    y=co2_mean, 
                    line_dash="dot", 
                    line_color="red", 
                    annotation_text=f"Moyenne CO2 ({co2_mean} g/km)", 
                    annotation_position="top right",
                    annotation=dict(bgcolor="rgba(255,255,255,0.8)")
                )
                st.plotly_chart(fig)
            else:

                # Sélection dynamique du nombre de catégories à afficher
                top_n = st.slider("**Sélectionnez le nombre de catégories à afficher :**", min_value=5, max_value=60, value=20, step=5)

                # Sélecteur pour le tri (ascendant = moins de CO2, descendant = plus de CO2)
                sort_order = st.radio(
                    "**Sélectionnez l'ordre de tri :**",
                    ["Ordre décroissant", "Ordre croissant"],
                    index=0  # Valeur par défaut : Plus de CO₂
                )

                # Définition l'ordre de tri en fonction de la sélection utilisateur
                ascending = True if sort_order == "Ordre croissant" else False

                # Calculer la moyenne du CO₂ par catégorie et trier
                avg_co2_by_var = (
                    df_nettoye.groupby(selected_feature)[target_variable]
                    .mean()
                    .round(2)
                    .reset_index()
                    .sort_values(by=target_variable, ascending=ascending)
                    .head(top_n)
                )

                # Choisir le dégradé de couleur selon le tri
                color_scale = ["green", "yellow", "orange", "red"] if not ascending else ["green", "yellow", "red"]

                # Création du graphique
                fig_hist = px.bar(
                    avg_co2_by_var,
                    x=selected_feature, 
                    y=target_variable,
                    title=f"Top {top_n} {selected_feature} émettant le {'plus' if not ascending else 'moins'} de CO2",
                    color=target_variable,
                    color_continuous_scale=color_scale,
                    text_auto=True
                )

                # Ajouter la ligne de moyenne
                fig_hist.add_hline(
                    y=co2_mean, 
                    line_dash="dot", 
                    line_color="red",                     
                    annotation_text=f"Moyenne CO2 ({co2_mean} g/km)", 
                    annotation_position="top right",
                    annotation=dict(bgcolor="rgba(255,255,255,0.8)")
                )

                # Incliner les labels des abscisses
                fig_hist.update_layout(xaxis_tickangle=-45)

                # Afficher le graphique dans Streamlit
                st.plotly_chart(fig_hist)


        else:
            st.warning("La variable 'CO2' n'a pas été trouvée dans le dataset.")
        
        st.subheader("Analyse des corrélations")
        numerical_columns = df_nettoye.select_dtypes(include='number').columns.tolist()
        corr_matrix = df_nettoye[numerical_columns].corr().round(2)
        fig_corr = px.imshow(
                            corr_matrix, 
                            text_auto=True, 
                            title="Matrice de corrélation des variables numériques", 
                            color_continuous_scale='reds')
        fig_corr.update_layout(
                                xaxis_tickangle=-45,
                                width=700,
                                height=700)
        st.plotly_chart(fig_corr)
    
    elif section == "🚨 Analyse des outliers":
        st.header("🚨 Analyse des outliers")

        # Sélection de la variable à analyser
        selected_variable = st.selectbox("Sélectionnez une variable :", df_nettoye.select_dtypes(include='number').columns)

        # Détection des outliers et affichage du boxplot
        if selected_variable:
            df_outliers, fig_outliers = detecter_outliers_plotly_var(df_nettoye[selected_variable])

            # Affichage du nombre d'outliers
            st.dataframe(df_outliers)

            # Affichage du boxplot
            st.plotly_chart(fig_outliers)

    elif section == "🤖 Modélisation":
        
        st.header("🤖 Modélisation des émissions de CO2")
        
        import os
        model_files = ["LinearRegression_model.pkl", "XGBoost_model.pkl", "Lasso_model.pkl", "DecisionTreeRegressor_model.pkl", "GradientBoostingRegressor_model.pkl", "RandomForestRegressor_model.pkl", "Ridge_model.pkl", "SGDRegressor_model.pkl"]  # Exemple de modèles enregistrés
        model_files = ["LinearRegression_model.pkl"] 
        
        selected_model_file = st.selectbox("Choisissez un modèle enregistré à charger:", model_files)
        
        try:
            model = joblib.load("models/" + selected_model_file)
            if isinstance(model, tuple):  # Vérifier si le modèle a été sauvegardé sous forme de tuple
                preprocessor, model = model  # Extraire le préprocesseur et le modèle
                X_test = df_nettoye.drop(columns=['CO2'])  # Supprime la colonne cible pour garder uniquement les features
                X_test_transformed = preprocessor.transform(X_test)
            elif hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                X_test_transformed = preprocessor.transform(X_test)
            else:
                X_test = df_nettoye.drop(columns=['CO2'])  # Supprime la colonne cible pour garder uniquement les features
                X_test_transformed = X_test
            
            X_test = df_nettoye.drop(columns=['CO2'])  # Supprime la colonne cible pour garder uniquement les features  # Charger les données de test
            y_test = df_nettoye['CO2']  # Sélectionne la colonne cible
            
            y_pred = model.predict(X_test_transformed)
            
            r2 = round(r2_score(y_test, y_pred), 2)
            mse = round(mean_squared_error(y_test, y_pred), 2)
            mae = round(mean_absolute_error(y_test, y_pred), 2)
            
            st.subheader("Performances du modèle chargé")
            st.write(f"**R2 Score** : {r2}")
            st.write(f"**MSE** : {mse}")
            st.write(f"**MAE** : {mae}")
            
        except FileNotFoundError:
            st.error("Le modèle sélectionné ou les données de test ne sont pas disponibles. Vérifiez les fichiers.")
        
        try:
            results = []
            for model_file in model_files:
                model = joblib.load("models/" + model_file)
                y_pred = model.predict(X_test)
                
                r2 = round(r2_score(y_test, y_pred), 5)
                mse = round(mean_squared_error(y_test, y_pred), 5)
                mae = round(mean_absolute_error(y_test, y_pred), 5)
                rmse = round(np.sqrt(mse), 5)
                ratio_rmse_mae = round(rmse - mae, 5) if mae != 0 else None
                
                results.append({"Modèle": model_file, "R2": r2, "MSE": mse, "RMSE": rmse, "MAE": mae, "RMSE - MAE": ratio_rmse_mae})
            
            
            df_models = pd.DataFrame(results)
            st.subheader("Comparaison des performances des modèles")
            st.dataframe(df_models)
            
            st.subheader("Top 3 meilleurs modèles")
            top_3_models = df_models.head(3)
            st.dataframe(top_3_models)
            # best_model = df_models.sort_values(by='R2', ascending=False).iloc[0]
            # st.write(f"**Modèle sélectionné** : {best_model['Modèle']}")
            # st.write(f"- **R2 Score** : {best_model['R2']}")
            # st.write(f"- **MSE** : {best_model['MSE']}")
            # st.write(f"- **MAE** : {best_model['MAE']}")
        except Exception as e:
            st.error(f"Erreur lors de la comparaison des modèles : {str(e)}")
    
    elif section == "🚀 Démo":
        st.header("🚀 Démo : prédiction des Émissions de CO2")

        # Charger le modèle de régression linéaire
        model_path = "models/LinearRegression_model.pkl"  # Chemin du modèle
        loaded_model = joblib.load(model_path)

        st.markdown("Entrez les caractéristiques du véhicule pour estimer les émissions de CO2 (g/km).")

        # Entrée des caractéristiques du véhicule
        masse_vide = st.number_input("⚖️ Masse à vide (kg)", min_value=500, max_value=3000, value=1845, step=50)
        carburant = st.selectbox("⛽ Type de carburant", ["essence", "diesel", "hybride", "e85", "gaz"])
        cylindree = st.number_input("🔧 Cylindrée moteur (cm³)", min_value=500, max_value=8000, value=1332, step=50)
        puissance = st.number_input("⚡ Puissance moteur (kW)", min_value=40, max_value=1200, value=96, step=5)
        conso = st.number_input("⛽ Consommation de carburant (L/100km)", min_value=0.4, max_value=30.0, value=2.0, step=0.1)

        # Convertir les entrées en DataFrame
        new_data = pd.DataFrame({
            "Masse à vide": [masse_vide],
            "Carburant": [carburant],
            "Cylindrée moteur": [cylindree],
            "Puissance moteur": [puissance],
            "Consommation carburant": [conso]
        })

        # Bouton de prédiction
        if st.button("📊 Prédire les émissions de CO₂"):
            # Faire la prédiction
            prediction = loaded_model.predict(new_data)
            
            valeur_co2 = prediction[0]

            def get_co2_class(co2_value):
                if co2_value <= 100:
                    return '<span class="co2-badge a">A</span> <span class="empreinte">(Empreinte carbone faible)</span>'
                elif co2_value <= 120:
                    return '<span class="co2-badge b">B</span> <span class="empreinte">(Empreinte carbone faible)</span>'
                elif co2_value <= 140:
                    return '<span class="co2-badge c">C</span> <span class="empreinte">(Empreinte carbone faible)</span>'
                elif co2_value <= 160:
                    return '<span class="co2-badge d">D</span> <span class="empreinte">(Empreinte carbone modérée)</span>'
                elif co2_value <= 200:
                    return '<span class="co2-badge e">E</span> <span class="empreinte">(Empreinte carbone modérée)</span>'
                elif co2_value <= 250:
                    return '<span class="co2-badge f">F</span> <span class="empreinte">(Empreinte carbone élevée)</span>'
                else:
                    return '<span class="co2-badge g">G</span> <span class="empreinte">(Empreinte carbone élevée)</span>'

            # CSS pour styliser les pastilles rondes
            st.markdown("""
                <style>
                .empreinte {
                    font-weight:200;        
                    font-size: 13px;
                }
                .co2-badge {
                    display: inline-block;
                    width: 30px;
                    height: 30px;
                    line-height: 30px;
                    text-align: center;
                    font-weight: bold;
                    color: white;
                    border-radius: 50%;
                    font-size: 16px;
                    margin-right: 10px;
                    margin-left: 10px;
                    
                }
                .a { background-color: #006400; } /* Vert foncé */
                .b { background-color: #008000; } /* Vert moyen */
                .c { background-color: #9ACD32; } /* Vert clair */
                .d { background-color: #FFD700; } /* Jaune */
                .e { background-color: #FFA500; } /* Orange */
                .f { background-color: #FF8C00; } /* Orange foncé */
                .g { background-color: #DC143C; } /* Rouge */
                </style>
            """, unsafe_allow_html=True)

            co2_class = get_co2_class(valeur_co2)

            # Affichage du résultat

            st.markdown(f"""
                <div style="background-color:#d4edda;padding:10px;border-radius:5px;color:#155724;font-weight:bold;">
                    🌱 Estimation des émissions de CO2 : <b>{valeur_co2:.2f} g/km</b> <br><br>
                    Classe d'émission CO2 : {co2_class}
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div style="background-color:#cce5ff;padding:10px;border-radius:5px;color:#155724;font-weight:bold; margin-top:10px">
                    ℹ️ Comment sont classés les véhicules ? <br><br>
                    <span class="co2-badge a">A</span> <= 100 g/km<br><br>
                    <span class="co2-badge b">B</span> > 100 g/km et <= 120 g/km<br><br>
                    <span class="co2-badge c">C</span> > 120 g/km et <= 140 g/km<br><br>
                    <span class="co2-badge d">D</span> > 140 g/km et <= 160 g/km<br><br>
                    <span class="co2-badge e">E</span> > 160 g/km et <= 200 g/km<br><br>
                    <span class="co2-badge f">F</span> > 200 g/km et <= 250 g/km<br><br>
                    <span class="co2-badge g">G</span> > 250 g/km
                    
                </div>
            """, unsafe_allow_html=True)

            #st.info("ℹ️ Comment sont classés les véhicules ?  \n <span class="co2-badge a">A</span> n Classe E :  \n Classe F :   \n Classe G : > 250 g/km")

    elif section == "📝 Conclusion":
        st.header("📝 Conclusion")
        st.subheader("Synthèse des résultats")
        st.write("Le modèle XX offre la meilleure précision.")
        st.subheader("Améliorations futures")
        st.markdown("""
                    - Ajout de nouvelles variables (aérodynamisme, type de transmission).
                    - Mise en production du modèle via une API pour prédire les émissions en temps réel.
                    """)


except FileNotFoundError:
    st.error("Fichiers non trouvés. Vérifiez que 'datas_nettoyees_model_FR.csv' est bien dans le même dossier que ce script.")
