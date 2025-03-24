import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import warnings
import pickle
from time import time

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline


warnings.filterwarnings('ignore')


class EnergyBenchmarkingAnalysis:
    """
    Classe pour charger, prétraiter et visualiser les données de benchmarking énergétique.
    """
    def __init__(self, file_2015, file_2016):
        """
        Initialisation avec les chemins d'accès aux fichiers CSV pour 2015 et 2016.
        """
        self.file_2015 = file_2015
        self.file_2016 = file_2016
        self.data_2015 = None
        self.data_2016 = None
        self.data2015 = None
        self.data1516 = None
        self.cleaned_data = None

    def load_data(self):
        """Charge les fichiers CSV pour 2015 et 2016."""
        self.data_2015 = pd.read_csv(self.file_2015)
        self.data_2016 = pd.read_csv(self.file_2016)
        print("Données chargées.")

    def preprocess_2015(self):
        """
        Prétraite le jeu de données de 2015 en extrayant la latitude et la longitude de la colonne 'Location'.
        Utilise ast.literal_eval pour évaluer en toute sécurité la chaîne de caractères de type JSON.
        """
        self.data2015 = pd.DataFrame()
        for idx, row in self.data_2015.iterrows():
            data_dict = ast.literal_eval(row['Location'])
            lat = data_dict.get('latitude')
            lon = data_dict.get('longitude')
            row = row.drop('Location')
            cols = list(row.index) + ['latitude', 'longitude']
            temp_df = pd.DataFrame([list(row) + [lat, lon]], columns=cols)
            self.data2015 = pd.concat([self.data2015, temp_df], ignore_index=True)
        print("Prétraitement des données 2015 effectué.")

    def merge_data(self):
        """
        Fusionne les jeux de données de 2015 et 2016.
        Aligne les colonnes cibles et les coordonnées géographiques.
        """
        data_EC2 = pd.concat([self.data2015["GHGEmissions(MetricTonsCO2e)"], self.data_2016["TotalGHGEmissions"]])
        Longitud = pd.concat([self.data2015["longitude"], self.data_2016["Longitude"]])
        Latitud = pd.concat([self.data2015["latitude"], self.data_2016["Latitude"]])
        self.data1516 = pd.concat([self.data2015, self.data_2016], ignore_index=True)
        self.data1516 = self.data1516.loc[:, ~self.data1516.columns.duplicated()]
        self.data1516["data_EC2"] = data_EC2.reset_index(drop=True)
        self.data1516["Longitud"] = Longitud.reset_index(drop=True)
        self.data1516["Latitud"] = Latitud.reset_index(drop=True)
        print("Fusion des données effectuée.")

    def clean_data(self):
        """
        Nettoie le jeu de données fusionné en :
          - Supprimant les colonnes ayant plus de 60% de valeurs manquantes.
          - Retirant une liste prédéfinie de colonnes inutiles.
          - Enlevant les lignes où 'NumberofBuildings' vaut 0.
          - Convertissant les colonnes de coordonnées en valeurs numériques.
          - Supprimant les colonnes dupliquées.
        """
        Moyenne_nulls = self.data1516.isnull().mean(axis=0)
        c = []
        for val in Moyenne_nulls:
            if val < 0.6:
                templist = list(Moyenne_nulls[Moyenne_nulls == val].index)
                c.extend(templist)
        consistants = list(set(c))
        data_1516 = self.data1516.loc[:, consistants]

        inutile = [
            "Neighborhood", "SPD Beats", "Seattle Police Department Micro Community Policing Plan Areas",
            "Latitude", "latitude", "ListOfAllPropertyUseTypes", "ComplianceStatus", "OtherFuelUse(kBtu)",
            "GHGEmissionsIntensity", "SiteEnergyUseWN(kBtu)", "State", "PrimaryPropertyType",
            "TaxParcelIdentificationNumber", "PropertyName", "Zip Codes", "ZipCode", "Longitude",
            "CouncilDistrictCode", "Neighborhood", "ListOfAllPropertyUseTypes", "LargestPropertyUseType",
            "PropertyGFABuilding(s)", "LargestPropertyUseTypeGFA", "ThirdLargestPropertyUseTypeGFA",
            "GHGEmissions(MetricTonsCO2e)" "SecondLargestPropertyUseType", "SecondLargestPropertyUseTypeGFA",
            "ThirdLargestPropertyUseType", "longitude", "NaturalGas(therms)", "DefaultData", "Comment",
            "ComplianceStatus", "Outlier", "2010 Census Tracts", "Address", "City Council Districts",
            "SourceEUIWN(kBtu/sf)", "SiteEUIWN(kBtu/sf)", "PropertyGFAParking",
            "GHGEmissionsIntensity(kgCO2e/ft2)", "SecondLargestPropertyUseType", "City",
            "GHGEmissions(MetricTonsCO2e)", "TotalGHGEmissions"
        ]
        for col in inutile:
            try:
                consistants.remove(col)
            except ValueError:
                pass
        data_1516 = data_1516.loc[:, consistants]
        idx_drop = data_1516[data_1516['NumberofBuildings'] == 0.0].index
        data_1516.drop(idx_drop, inplace=True)
        data_1516['Latitud'] = pd.to_numeric(data_1516['Latitud'], errors='coerce')
        data_1516['Longitud'] = pd.to_numeric(data_1516['Longitud'], errors='coerce')
        data_1516 = data_1516.loc[:, ~data_1516.columns.duplicated()]
        data_1516 = data_1516.reset_index(drop=True)
        self.cleaned_data = data_1516.copy()
        print("Nettoyage des données terminé. Forme des données nettoyées :", self.cleaned_data.shape)

    def add_derived_features(self):
        """
        Crée des variables dérivées telles que data_TCEnergy et data_EmissionsCO2.
        Cette méthode doit être appelée après le nettoyage des données.
        """
        if "SiteEnergyUse(kBtu)" in self.cleaned_data.columns:
            self.cleaned_data["data_TCEnergy"] = self.cleaned_data["SiteEnergyUse(kBtu)"]
        else:
            print("La colonne 'SiteEnergyUse(kBtu)' est introuvable dans les données nettoyées.")
        if "data_EC2" in self.cleaned_data.columns:
            self.cleaned_data["data_EmissionsCO2"] = self.cleaned_data["data_EC2"]
        else:
            print("La colonne 'data_EC2' est introuvable dans les données nettoyées.")

    def visualize_missing_values(self):
        """
        Visualise le pourcentage de valeurs non nulles pour chaque caractéristique.
        """
        plt.figure(figsize=(10, 5))
        non_null_pct = (self.cleaned_data.notnull().mean(axis=0) * 100)
        non_null_pct.plot.barh(color="red")
        plt.xlim(0, 100)
        plt.title("Pourcentage de valeurs non nulles")
        plt.xlabel("Pourcentage")
        plt.ylabel("Caractéristiques")
        plt.show()

    def boxplot_univ(self, feature, plotColor="#CC9900", ylim=None):
        """
        Trace un boxplot univarié pour une caractéristique donnée.
        
        Paramètres :
            - feature : nom de la caractéristique.
            - plotColor : couleur du boxplot.
            - ylim : limites de l'axe y (facultatif).
        """
        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 3))
        sns.boxplot(data=self.cleaned_data, y=feature, color=plotColor)
        if ylim:
            plt.ylim(ylim)
        plt.title(f"Boxplot pour {feature}")
        plt.show()

    def boxplot_multiv(self, feature, plotColor="#CC9900", ylim=None):
        """
        Trace un boxplot bivarié pour une caractéristique donnée, groupé par 'BuildingType'.
        
        Paramètres :
            - feature : nom de la caractéristique.
            - plotColor : couleur du boxplot.
            - ylim : tuple (min, max) pour l'axe y (facultatif).
        """
        plt.figure(figsize=(15, 4))
        sns.set_style("whitegrid")
        sns.boxplot(data=self.cleaned_data, x="BuildingType", y=feature, color=plotColor)
        if ylim:
            plt.ylim(ylim)
        plt.title(f"Boxplot de {feature} par type de bâtiment")
        plt.show()

    def plot_buildingtype_distribution(self):
        """
        Affiche un camembert pour la distribution des types de bâtiments et un histogramme pour ENERGYSTARScore.
        """
        data_cl = self.cleaned_data['BuildingType'].value_counts(normalize=True)
        plt.figure(figsize=(7, 7))
        plt.pie(data_cl.values, labels=data_cl.index, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title("Répartition des bâtiments selon leur type")
        plt.show()
        plt.figure()
        plt.hist(self.cleaned_data['ENERGYSTARScore'].dropna(), bins=100, edgecolor='k')
        plt.xlabel('ENERGYSTARScore')
        plt.ylabel('Nombre de bâtiments')
        plt.title('Distribution du score Energy Star')
        plt.show()

    def correlation_heatmap(self, features=None, title="Correlation Heatmap"):
        """
        Affiche une carte de chaleur de la matrice de corrélation pour les caractéristiques sélectionnées.
        """
        if features is None:
            features = self.cleaned_data.columns
        corr = self.cleaned_data[features].corr().round(1)
        plt.figure(figsize=(10, 9))
        plt.title(title)
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        sns.set(font_scale=1.2)
        with sns.axes_style("white"):
            sns.heatmap(corr, annot=True, vmax=1, cmap="RdBu_r", square=True, mask=mask)
        plt.show()

    def create_dummies(self):
        """
        Convertit la variable catégorielle 'BuildingType' en variables fictives (dummies).
        """
        dummies = pd.get_dummies(self.cleaned_data['BuildingType'], prefix='BuildingType')
        self.cleaned_data = pd.concat([self.cleaned_data, dummies], axis=1)
        print("Variables fictives créées.")

    def save_clean_data(self, filename="cleaned_1516.parquet"):
        """Enregistre les données nettoyées dans un fichier Parquet."""
        self.cleaned_data.to_parquet(filename, index=False)
        print(f"Données nettoyées enregistrées dans {filename}.")

    def load_clean_data(self, filename="cleaned_1516.parquet"):
        """Charge les données nettoyées depuis un fichier Parquet."""
        self.cleaned_data = pd.read_parquet(filename)
        print("Données nettoyées chargées.")

    def get_subset_for_modeling(self, target_feature, drop_features=[]):
        """
        Crée un sous-ensemble des données pour la modélisation.
        
        Paramètres :
            - target_feature (str) : nom de la variable cible.
            - drop_features (list) : liste des noms de colonnes à supprimer.
        
        Retourne :
            - X (DataFrame) : caractéristiques numériques.
            - y (Series) : variable cible.
        """
        data_copy = self.cleaned_data.copy()
        data_copy = data_copy.dropna(subset=[target_feature])
        for col in drop_features:
            if col in data_copy.columns:
                data_copy = data_copy.drop(columns=col)
        y = data_copy[target_feature]
        X = data_copy.select_dtypes(include=[np.number]).drop(columns=[target_feature])
        return X, y

    def split_and_scale(self, X, y, test_size=0.3, random_state=42):
        """
        Sépare les données en ensembles d'entraînement et de test, impute les valeurs manquantes et standardise les caractéristiques.
        
        Retourne :
            x_train, x_test, y_train, y_test, x_train_std, x_test_std.
        """
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        imputer = SimpleImputer(strategy='mean')
        x_train = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
        x_test = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns, index=x_test.index)
        scaler = StandardScaler().fit(x_train)
        x_train_std = scaler.transform(x_train)
        x_test_std = scaler.transform(x_test)
        return x_train, x_test, y_train, y_test, x_train_std, x_test_std


class EnergyModeling:
    """
    Classe pour entraîner et évaluer différents modèles de régression.
    """
    def __init__(self, x_train_std, x_test_std, y_train, y_test, feature_names):
        self.x_train_std = x_train_std
        self.x_test_std = x_test_std
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names

    def linear_regression(self):
        """Entraîne et évalue un modèle de régression linéaire."""
        model = LinearRegression()
        start = time()
        model.fit(self.x_train_std, self.y_train)
        duration = time() - start
        train_pred = model.predict(self.x_train_std)
        test_pred = model.predict(self.x_test_std)
        print(f"Régression linéaire entraînée en {duration:.3f}s")
        for name, coef in zip(self.feature_names, model.coef_):
            print(f"{name}: {coef}")
        print("RMSE (train) :", np.sqrt(mean_squared_error(self.y_train, train_pred)))
        print("R2 (train) :", r2_score(self.y_train, train_pred))
        print("RMSE (test)  :", np.sqrt(mean_squared_error(self.y_test, test_pred)))
        print("R2 (test)  :", r2_score(self.y_test, test_pred))
        return model

    def ridge_regression(self, alpha=0.1):
        """Entraîne et évalue un modèle de régression Ridge."""
        model = Ridge(alpha=alpha)
        start = time()
        model.fit(self.x_train_std, self.y_train)
        duration = time() - start
        train_pred = model.predict(self.x_train_std)
        test_pred = model.predict(self.x_test_std)
        print(f"Régression Ridge (alpha={alpha}) entraînée en {duration:.3f}s")
        for name, coef in zip(self.feature_names, model.coef_):
            print(f"{name}: {coef}")
        print("RMSE (train) :", np.sqrt(mean_squared_error(self.y_train, train_pred)))
        print("R2 (train) :", r2_score(self.y_train, train_pred))
        print("RMSE (test)  :", np.sqrt(mean_squared_error(self.y_test, test_pred)))
        print("R2 (test)  :", r2_score(self.y_test, test_pred))
        return model

    def ridge_regression_cv(self, alphas):
        """Réalise une régression Ridge avec validation croisée pour le réglage des hyperparamètres."""
        model = Ridge()
        param_grid = {'alpha': alphas}
        grid = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
        start = time()
        grid.fit(self.x_train_std, self.y_train)
        duration = time() - start
        best_alpha = grid.best_params_['alpha']
        print(f"Meilleur alpha pour Ridge CV : {best_alpha}, entraîné en {duration:.3f}s")
        best_model = grid.best_estimator_
        train_pred = best_model.predict(self.x_train_std)
        test_pred = best_model.predict(self.x_test_std)
        print("RMSE (train) :", np.sqrt(mean_squared_error(self.y_train, train_pred)))
        print("R2 (train) :", r2_score(self.y_train, train_pred))
        print("RMSE (test)  :", np.sqrt(mean_squared_error(self.y_test, test_pred)))
        print("R2 (test)  :", r2_score(self.y_test, test_pred))
        return best_model

    def kernel_ridge(self, alpha=1.0, gamma=0.01):
        """Entraîne et évalue un modèle de régression Kernel Ridge."""
        model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
        start = time()
        model.fit(self.x_train_std, self.y_train)
        duration = time() - start
        test_pred = model.predict(self.x_test_std)
        rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        r2 = r2_score(self.y_test, test_pred)
        print(f"Kernel Ridge Regression entraîné en {duration:.3f}s")
        print("RMSE (test) :", rmse)
        print("R2 (test) :", r2)
        return model

    def svr_model(self):
        """Entraîne et évalue un modèle de Support Vector Regression (SVR) avec GridSearchCV."""
        param_grid = {"C": [1e0, 1e1, 1e2],
                      "gamma": np.logspace(-2, 2, 3)}
        model = SVR(kernel='rbf', gamma=0.01)
        grid = GridSearchCV(model, param_grid, cv=5)
        start = time()
        grid.fit(self.x_train_std, self.y_train)
        duration = time() - start
        best_model = grid.best_estimator_
        test_pred = best_model.predict(self.x_test_std)
        rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        r2 = r2_score(self.y_test, test_pred)
        print(f"SVR entraîné en {duration:.3f}s avec les meilleurs paramètres : {grid.best_params_}")
        print("RMSE (test) :", rmse)
        print("R2 (test) :", r2)
        return best_model

    def mlp_regressor(self):
        """Entraîne et évalue un modèle MLPRegressor à l'aide d'un pipeline."""
        pipeline = make_pipeline(QuantileTransformer(),
                                 MLPRegressor(hidden_layer_sizes=(50, 50),
                                              learning_rate_init=0.01,
                                              early_stopping=True))
        start = time()
        pipeline.fit(self.x_train_std, self.y_train)
        duration = time() - start
        test_pred = pipeline.predict(self.x_test_std)
        rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        r2 = r2_score(self.y_test, test_pred)
        print(f"MLPRegressor entraîné en {duration:.3f}s")
        print("RMSE (test) :", rmse)
        print("R2 (test) :", r2)
        return pipeline

    def gradient_boosting(self, param_grid=None):
        """
        Entraîne et évalue un modèle de Gradient Boosting Regressor à l'aide de GridSearchCV.
        Si aucune grille n'est fournie, une grille par défaut est utilisée.
        """
        if param_grid is None:
            param_grid = {"n_estimators": [50, 100, 125, 150, 200],
                          "max_depth": [3, 5, 7],
                          "loss": ["ls", "lad", "huber", "quantile"]}
        model = GradientBoostingRegressor()
        grid = GridSearchCV(model, param_grid, scoring="r2", cv=5, n_jobs=-1)
        start = time()
        grid.fit(self.x_train_std, self.y_train)
        duration = time() - start
        best_model = grid.best_estimator_
        test_pred = best_model.predict(self.x_test_std)
        rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        r2 = r2_score(self.y_test, test_pred)
        print(f"Gradient Boosting Regressor entraîné en {duration:.3f}s")
        print("RMSE (test) :", rmse)
        print("R2 (test) :", r2)
        print("Meilleur modèle :", best_model)
        print("Importance des caractéristiques :")
        for name, imp in zip(self.feature_names, best_model.feature_importances_):
            print(f"{name}: {imp}")
        return best_model


# ----------------- Programme principal -----------------
if __name__ == "__main__":
    # Initialisation de la classe d'analyse avec les chemins d'accès aux fichiers CSV
    analysis = EnergyBenchmarkingAnalysis("2015-building-energy-benchmarking.csv",
                                          "2016-building-energy-benchmarking.csv")
    analysis.load_data()              # Charge les données
    analysis.preprocess_2015()         # Prétraite les données de 2015
    analysis.merge_data()              # Fusionne les données de 2015 et 2016
    analysis.clean_data()              # Nettoie les données fusionnées
    analysis.add_derived_features()    # Ajoute les variables dérivées 'data_TCEnergy' et 'data_EmissionsCO2'

    # Visualisations
    analysis.visualize_missing_values()  # Affiche le pourcentage de valeurs non nulles
    analysis.boxplot_univ("Electricity(kBtu)", plotColor="#F5F5DC", ylim=(0, 10000000))
    analysis.boxplot_univ("NaturalGas(kBtu)", plotColor="#33CC33", ylim=(0, 4000000))
    analysis.plot_buildingtype_distribution()  # Affiche la distribution des types de bâtiments
    analysis.correlation_heatmap(features=["SiteEnergyUse(kBtu)", "data_EC2", "ENERGYSTARScore"],
                                 title="Corrélation entre énergie et émissions")
    analysis.create_dummies()
    analysis.save_clean_data("cleaned_1516.parquet")

    # ----------------- Utilisation de la nouvelle version améliorée de boxplot_multiv -----------------
    # Tracé du boxplot pour "Electricity(kBtu)" avec une limite y de (0, 50000000)
    analysis.boxplot_multiv("Electricity(kBtu)", "#F5F5DC", ylim=(0, 50000000))
    # Tracé du boxplot pour "NaturalGas(kBtu)" avec une limite y de (0, 20000000)
    analysis.boxplot_multiv("NaturalGas(kBtu)", "#33CC33", ylim=(0, 20000000))

    # ----------------- Modélisation pour la cible data_TCEnergy -----------------
    X, y = analysis.get_subset_for_modeling(target_feature="data_TCEnergy")
    x_train, x_test, y_train, y_test, x_train_std, x_test_std = analysis.split_and_scale(X, y, test_size=0.2, random_state=42)
    modeler = EnergyModeling(x_train_std, x_test_std, y_train, y_test, feature_names=x_train.columns)
    # On entraîne plusieurs modèles et ici nous choisissons par exemple le Gradient Boosting comme meilleur modèle
    lr_model = modeler.linear_regression()
    ridge_model = modeler.ridge_regression(alpha=0.1)
    ridge_cv_model = modeler.ridge_regression_cv(alphas=[1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 30, 50, 100, 200])
    kernel_ridge_model = modeler.kernel_ridge(alpha=1.0, gamma=0.01)
    svr_model = modeler.svr_model()
    mlp_model = modeler.mlp_regressor()
    gbr_model = modeler.gradient_boosting()

    # ----------------- Modélisation pour la cible data_EmissionsCO2 -----------------
    X2, y2 = analysis.get_subset_for_modeling(target_feature="data_EmissionsCO2")
    x2_train, x2_test, y2_train, y2_test, x2_train_std, x2_test_std = analysis.split_and_scale(X2, y2, test_size=0.2, random_state=42)
    modeler2 = EnergyModeling(x2_train_std, x2_test_std, y2_train, y2_test, feature_names=x2_train.columns)
    lr_model2 = modeler2.linear_regression()
    gbr_model2 = modeler2.gradient_boosting()

    # ----------------- Sauvegarde des deux meilleurs modèles dans des fichiers .pkl -----------------
    
    scaler_TCEnergy = StandardScaler().fit(x_train)  # Ajusté sur les données d'entraînement non standardisées
    pipeline_TCEnergy = make_pipeline(scaler_TCEnergy, gbr_model)
    with open("best_model_data_TCEnergy.pkl", "wb") as f:
        pickle.dump(pipeline_TCEnergy, f)
    print("Modèle pour data_TCEnergy sauvegardé dans best_model_data_TCEnergy.pkl")

    scaler_EC2 = StandardScaler().fit(x2_train)
    pipeline_EC2 = make_pipeline(scaler_EC2, gbr_model2)
    with open("best_model_data_EC2.pkl", "wb") as f:
        pickle.dump(pipeline_EC2, f)
    print("Modèle pour data_EC2 sauvegardé dans best_model_data_EC2.pkl")
