import base64
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently.report import Report
from sklearn import metrics
from tableone import TableOne

from CounterFactualType import CounterFactualType
from Counterfactuals.counterfactuals import (
    KDTreeCounterFactual,
    GeneticCounterfactual,
)
from SHAP.SHAP_functions import SHAPValues, SHAPType
from TypesOfVisualisationMethod import (
    CorrMatrixMethod,
    ImportanceType,
    OtherVisualisation,
)


class AI_API:

    """ Class die alle functionaliteiten van de AI modellen bevat
    dit bevat onder andere het inladen van het model, het voorspellen van de uitkomst, het genereren van counterfactuals en het genereren van SHAP visualisaties

    Attributes:
    - model: Het model dat wordt gebruikt voor de voorspellingen
    - model_path: Het pad naar het model bestand
    - shap_values: De SHAP Class die de SHAP waarden en visualisaties bevat
    - random_cf: De counterfactual class die de random counterfactuals genereert
    - genetic_cf: De counterfactual class die de genetic counterfactuals genereert
    - KDTree_cf: De counterfactual class die de KDTree counterfactuals genereert
    - data: De data die wordt gebruikt voor de counterfactuals
    - dummy_groupings: De dummy groupings die worden gebruikt voor de counterfactuals
    - use_feats: De features die worden gebruikt voor de counterfactuals

    """

    def __init__(self):
        self.model = None

        self.model_path = "./Modellen/xgb.pkl"
        self.load_model(self.model_path)

        self.shap_values = None

        self.random_cf = None
        self.genetic_cf = None
        self.KDTree_cf = None
        self.data = self.getData()

        self.dummy_groupings = {
            "race": ["AISAN", "BLACK", "HISPANIC", "OTHER", "WHITE", "unknown"],
            "first_careunit": [
                "CCU",
                "CVICU",
                "MICU",
                "MICU/SICU",
                "NICU",
                "SICU",
                "TSICU",
            ],
        }

        self.use_feats = ["age", "weight", "temperature", "gcs"]
        self.setup_counterfactuals(self.data, self.dummy_groupings, self.use_feats)
        self.setup_shap_values(self.data)

    def getPatientsWithSearchvalues(self, searchvalues):
        dataFrame = self.data.copy()

        # Filter out the None values
        toRemove = []
        del searchvalues["SAD"]
        for value in searchvalues:
            if searchvalues[value] == None:
                toRemove.append(value)

        # Remove the None values
        for value in toRemove:
            del searchvalues[value]

        # Filter the data
        for key in searchvalues:
            if key == "ETHNICITY" or key == "ICU":
                dataFrame = dataFrame[dataFrame[searchvalues[key]] == 1]
            elif searchvalues[key] == True:
                dataFrame = dataFrame[dataFrame[key] == 1]
            elif searchvalues[key] == False:
                dataFrame = dataFrame[dataFrame[key] == 0]
            elif key == 'GENDER':
                if searchvalues[key] == 'MALE':
                    dataFrame = dataFrame[dataFrame["gender"] == 1]
                else:
                    dataFrame = dataFrame[dataFrame["gender"] == 0]
            elif key == "AGE" or key == "WEIGHT":
                dataFrame = dataFrame[
                    dataFrame[key.lower()].between(
                        searchvalues[key][0], searchvalues[key][1]
                    )
                ]
            else:
                dataFrame = dataFrame[dataFrame[key] == searchvalues[key]]

        return dataFrame



    def getData(self) -> pd.DataFrame:
        """ Laads de data in vanuit een DTA bestand en zet deze om naar een pandas dataframe
        
        Returns:
            pd.dataframe: Ingeladen data
        """
        mod_path = Path(__file__).parent
        data_raw = pd.read_stata((mod_path / "Modellen/MIMIC-IV.dta").resolve())
        # BELANGRIJK: in verband met hoe de verschillende explainers werken, moet de data zo aangepast worden
        # Data variabele wordt gebruikt in standard API acties
        # Data_cf wordt gebruikt in counterfactuals
        data_cf = data_raw.drop(
            [
                "deliriumtime",
                "hosp_mort",
                "icu28dmort",
                "stay_id",
                "icustay",
                "hospstay",
                "sepsistime",
            ],
            axis=1,
        ).dropna()
        dummies = pd.get_dummies(data_cf["race"])
        data = data_cf.drop("race", axis=1).join(dummies)
        dummies = pd.get_dummies(data["first_careunit"])
        data = data.drop("first_careunit", axis=1).join(dummies)
        data["ID"] = data.reset_index(drop=True).index
        data = data.drop(["sad"], axis=1)
        self.data = data
        return data

    def predict(self, x) -> float:

        """ Voorspelt de uitkomst van het model op basis van de input x

        Args: x (str): De input waarop het model een voorspelling moet doen. De input moet een json dict zijn.

        Returns:
            Float: retuneeert de voorspelling van het model. een waarde van boven 0.5 is een positieve voorspelling en een waarde van onder 0.5 is een negatieve voorspelling
        """
        x = self.parse_inputs(x)

        # temp solution for XGBoost
        if isinstance(self.model, xgb.core.Booster):
            print(x)
            x = xgb.DMatrix(x)

            return self.model.predict(x)[0]

        return self.model.predict()[0][0]

    def load_model(self, path_model_file: str) -> bool:

        """Laad het model in vanuit een pickle bestand en zet deze in de model variabele
        
        Args: 
            path_model_file (str): Het pad naar het pickle bestand waar het model in staat

        Raises:
            e: Exception die wordt opgegooid als het inladen van het model niet lukt

        Returns:
            Bool: True als het inladen van het model is gelukt, anders False
        """
        try:
            # TODO: loading new model based on path_model_file that is in a seperate folder and is saved with pickle
            path_model_file = Path(__file__).parent / path_model_file
            self.model = joblib.load(path_model_file)
            return True
        except Exception as e:
            raise e

    def setup_counterfactuals(self, data, dummy_groupings, use_feats) -> bool:

        """Setup de counterfactuals voor de verschillende methodes

        Args:
            data (pd.DataFrame): De data die gebruikt wordt voor de counterfactuals
            dummy_groupings (dict): De dummy groupings die gebruikt worden voor de counterfactuals
            use_feats (list): De features die gebruikt worden voor de counterfactuals
        Returns:
            Bool: True als het opzetten van de counterfactuals is gelukt, anders False
        """

        if self.model is None:
            return False

        if "ID" in data.columns:
            data = data.drop("ID", axis=1)

        if isinstance(self.model, xgb.core.Booster):
            self.random_cf = GeneticCounterfactual(
                data=data,
                model=self.model,
                dummy_groupings=dummy_groupings,
                use_feats=use_feats,
                limit=3,
                population_size=data.shape[0],
            )
            self.genetic_cf = GeneticCounterfactual(
                data=data,
                model=self.model,
                dummy_groupings=dummy_groupings,
                limit=10,
                population_size=data.shape[0],
            )
            self.KDTree_cf = KDTreeCounterFactual(data, self.model)
            return True
        else:
            # TODO: case for sklearn models Return False for now
            return False
        return False

    def setup_shap_values(self, data) -> bool:

        """Setup de SHAP values van de data en het model. 

        Args: 
            Data (pd.DataFrame): De data die gebruikt wordt voor de SHAP values

        Raises:
            ValueError: Exception die wordt opgegooid als het model nog niet is ingeladen
            e: Exception die wordt opgegooid als het inladen van de SHAP values niet lukt

        Returns:
            Bool: True als het opzetten van de SHAP values is gelukt, anders False
        """

        if self.model is None:
            raise ValueError("Model is not loaded yet")
            return False
        try:
            if "ID" in data.columns:
                data = data.drop("ID", axis=1)

            self.shap_values = SHAPValues(data, self.model)
            self.shap_values.setup_shap_values()
            print("test")
            print(self.shap_values)
            return True
        except Exception as e:
            raise e


    def parse_inputs(self, x: str) -> pd.DataFrame:

        """Parse de input x naar een pandas dataframe

        Args: 
            x (str): De input die geparsed moet worden naar een pandas dataframe. Data kan binnenkomen als een json string.

        Returns:
            pd.Dataframe: De input x als pandas dataframe
        """

        # if os.path.isfile(x):
        #     x = json.load(open(x))
        # else:
        #     x = json.loads(x)
        return pd.DataFrame(x)

    def check_inputs(self, x: pd.DataFrame) -> bool:
        """Check of de input x compatible is met het model. Deze functie wordt op dit moment niet gebruikt. 
        Voor wanneer er verschillende modellen worden gebruikt is het handig om te checken of de input compatible is met het model

        Args: x (pd.DataFrame): De input die gecheckt moet worden

        Raises:
            ValueError: Exception die wordt opgegooid als de input niet compatible is met het model

        Returns:
            Bool: True als de input compatible is met het model, anders False
        """

        input_features = x.columns
        if isinstance(self.model, xgb.core.Booster):
            # XGBoost models
            model_features = self.model.get_booster().feature_names
            if not all([feature in model_features for feature in input_features]):
                raise ValueError("Input features are not compatible with the model")
            else:
                return True

        else:
            # SkLearn models
            # model_features = self.model.columns
            return False
        return False

    def get_counterfactual(
        self, input_df: pd.DataFrame, method=CounterFactualType.RANDOM, amount_of_cfs=2
    ) -> pd.DataFrame:

        """Genereert een counterfactual voor de input data

        Args:
            input_df (pd.DataFrame): De input data waarvoor een counterfactual moet worden gegenereerd (Moet compatible zijn met het model)
            method (CounterFactualType): De methode die gebruikt moet worden voor het genereren van de counterfactual (RANDOM, GENETIC, KDTREE)
            amount_of_cfs (int): Het aantal counterfactuals dat gegenereerd moet worden

        Raises:
            ValueError: Exception die wordt opgegooid als de methode niet gevonden kan worden

        Returns:
            pd.dataframe: De gegenereerde counterfactuals
        """

        mod_path = Path(__file__).parent

        # TODO: de counterfactuals werkt nu niet. De match case werkt wel!
        match method.name:
            case CounterFactualType.RANDOM.name:
                chosen_method = self.random_cf
            case CounterFactualType.GENETIC.name:
                chosen_method = self.genetic_cf
            case CounterFactualType.KDTREE.name:
                chosen_method = self.KDTree_cf
            case _:
                raise ValueError("Method: " + method + " not found")

        input_df = self.parse_inputs(input_df)
        print(input_df)
        # print(input_df['sad'])
        # input_df = input_df.drop("ID", axis=1)
        cf_output = chosen_method.generate(input_df, amount_of_cfs)
        return cf_output

    def get_shap_visualization(
        self,
        type_visualization: SHAPType,
        range: tuple = None,
        feature_name: str = None,
        interaction_index: str = None,
        index: int = None,
        max_display: int = None,
        list_indices: list = None,
    ) -> any:
        return self.shap_values.generate(
            self.data,
            type_visualization,
            max_display,
            range,
            feature_name,
            interaction_index,
            index,
            list_indices,
        )

    def predict_probability(self, data: pd.DataFrame) -> float:

        """
        Berekent de voorspelde zekerheid van het model op basis van de input data

        Args: 
            data (pd.Dataframe): input data die ook gebruikt wordt voor de voorspelling

        Returns:
            float: de voorspelde zekerheid van het model
        """

        if self.model is None:
            return False

        data = self.parse_inputs(data)

        # just use predict as a placeholder for now
        # XGBoost models do not have a predict_proba method like SKLearn models
        # The documentation says it has a predict proba method for classifiers but it only works in a research envriorment when we were training the model
        # and this model we have is a binary classifier model. its output is the probability of the positive class
        # when we implement a Sklearn model we can use the predict_proba method. but not for now
        if isinstance(self.model, xgb.core.Booster):
            data = xgb.DMatrix(data)
            return self.model.predict(data)[0]

        return False

    def generate_table1(self, data):
        pass

    def get_similarity(self, index, amount, filterData):

        similar_df = self.data.iloc[
                         np.argsort(metrics.pairwise_distances(self.data, Y=self.data.iloc[[index]]).flatten())
                     ][1:amount+1]

        missing_features = ['age', 'weight', 'temperature', 'AISAN', 'mbp', 'CVICU', 'seda', 'stroke', 'BLACK', 'sbp', 'hyperte', 'cr',
                            'ptt', 'Mg', 'spo2', 'NICU', 'Cl', 'vaso', 'aniongap', 'bun', 'vent', 'dm', 'crrt',
                            'resp_rate', 'bicarbonate', 'ami', 'inr', 'copd', 'wbc', 'CCU', 'Ca', 'Na', 'ckd',
                            'HISPANIC', 'unknown', 'gender', 'P', 'hemoglobin', 'pt', 'aki', 'glu', 'OTHER', 'gcs',
                            'platelet', 'TSICU', 'heart_rate', 'sofa_score', 'MICU', 'dbp', 'WHITE', 'SICU',
                            'MICU/SICU', 'K']


        if(filterData is not None or filterData == []):
            missing_features_filtered = [feature for feature in missing_features if feature not in filterData]
            for filter in filterData:
                if (filter == 'ethnicity'):
                    missing_features_filtered.remove('AISAN')
                    missing_features_filtered.remove('BLACK')
                    missing_features_filtered.remove('HISPANIC')
                    missing_features_filtered.remove('OTHER')
                    missing_features_filtered.remove('WHITE')
                    missing_features_filtered.remove('unknown')
                elif (filter == 'icu'):
                    missing_features_filtered.remove('CCU')
                    missing_features_filtered.remove('CVICU')
                    missing_features_filtered.remove('MICU')
                    missing_features_filtered.remove('MICU/SICU')
                    missing_features_filtered.remove('NICU')
                    missing_features_filtered.remove('SICU')
                    missing_features_filtered.remove('TSICU')
            for feature in missing_features_filtered:
                similar_df[feature] = 0

        similar_df = similar_df.drop(columns="ID", axis=1)
        pred = self.model.predict(xgb.DMatrix(similar_df))
        pred = pred.astype(float)

        if(filterData is not None or filterData == []):
            for feature in missing_features_filtered:
                similar_df = similar_df.drop(columns=feature, axis=1)

        similar_dict = {
            "patients": [row.to_dict() for _, row in similar_df.iterrows()],
            "predictions": list(pred),
            "confidence": list(
                np.abs(np.invert(pred.round().astype(bool)).astype(float) - pred) * 100
            ),
        }

        return similar_dict

    def get_similarity_maar_dan_beter(self, index, amount, filterData):
        # TODO: iets minder wonky functie naam

        # TODO: (voor AI'ers) dit omzetten naar een dict en het zo versturen
        #  is echt bizar omslachtig; de pred scores hier berekenen is
        #  uberhaubt een slecht idee dit staat hier nu zo omdat er geen tijd
        #  meer voor is dit nu op te lossen, maar voor een volgende groep:
        #  maak het jezelf makkelijk en stuur data rond in een dataframe.
        #  het berekenen van pred kan ook plaatselijk gebeuren of in een
        #  aparte functie, dat had echt enorm veel gekut voorkomen

        if filterData is not None:

            if 'ethnicity' in filterData:
                filterData.extend(
                    ['AISAN', 'BLACK', 'HISPANIC', 'OTHER', 'WHITE', 'unknown'])
                filterData.remove('ethnicity')
            if 'icu' in filterData:
                filterData.extend(
                    ['CCU', 'CVICU', 'MICU', 'MICU/SICU', 'NICU', 'SICU', 'TSICU'])
                filterData.remove('icu')

            filterdata = self.data[filterData]
        else:
            filterdata = self.data
        i_sort = np.argsort(metrics.pairwise_distances(
            filterdata, Y=filterdata.iloc[[index]]
        ).flatten())

        res = self.data.iloc[i_sort][1:amount + 1]
        pred = self.model.predict(xgb.DMatrix(res.drop(columns="ID", axis=1)))
        pred = pred.astype(float)

        similar_dict = {
            "patients": [row.to_dict() for _, row in res.iterrows()],
            "predictions": list(pred),
            "confidence": list(
                np.abs(np.invert(pred.round().astype(bool)).astype(
                    float) - pred) * 100
            )
        }  # wat is deze dict format ook? waarom zit 'confidence' hierin?
           # verander dit asjeblieft; ik krijg een aneurysme

        return similar_dict

    def prepared_raw_data(self):
        mod_path = Path(__file__).parent
        data_raw = pd.read_stata((mod_path / "Modellen/MIMIC-IV.dta").resolve())

        data_table = data_raw.drop("stay_id", axis=1)
        data_table["gender"] = data_table["gender"].replace(
            to_replace=0.0, value="FEMALE"
        )
        data_table["gender"] = data_table["gender"].replace(
            to_replace=1.0, value="MALE"
        )

        data_table["sad"] = data_table["sad"].replace(to_replace=0.0, value="NON-SAD")
        data_table["sad"] = data_table["sad"].replace(to_replace=1.0, value="SAD")

        for i in [
            "vent",
            "crrt",
            "vaso",
            "seda",
            "ami",
            "ckd",
            "copd",
            "hyperte",
            "dm",
            "aki",
            "stroke",
            "hosp_mort",
        ]:
            data_table[i] = data_table[i].replace(to_replace=0.0, value="FALSE")
            data_table[i] = data_table[i].replace(to_replace=1.0, value="TRUE")

        data_table_train = pd.read_stata(
            (mod_path / "Modellen/train_set_raw.dta")
        ).drop("stay_id", axis=1)
        data_table_train["gender"] = data_table_train["gender"].replace(
            to_replace=0.0,
            value="FEMALE",
        )
        data_table_train["gender"] = data_table_train["gender"].replace(
            to_replace=1.0, value="MALE"
        )

        data_table_train["sad"] = data_table_train["sad"].replace(
            to_replace=0.0, value="NON-SAD"
        )
        data_table_train["sad"] = data_table_train["sad"].replace(
            to_replace=1.0, value="SAD"
        )

        for i in [
            "vent",
            "crrt",
            "vaso",
            "seda",
            "ami",
            "ckd",
            "copd",
            "hyperte",
            "dm",
            "aki",
            "stroke",
            "hosp_mort",
        ]:
            data_table_train[i] = data_table_train[i].replace(
                to_replace=0.0, value="FALSE"
            )
            data_table_train[i] = data_table_train[i].replace(
                to_replace=1.0, value="TRUE"
            )

        return (data_table, data_table_train)

    def other_visualization(self, type_visualization: OtherVisualisation, **kwargs):

        match type_visualization:
            case OtherVisualisation.TABLEONE.name:

                return {"type": "html", "content": self.get_tableone()}

            case OtherVisualisation.CORR_MATRIX.name:

                return {"type": "image", "content": self.get_corr_matrix()}
            case OtherVisualisation.CONFUSION_MATRIX.name:
                return {"type": "image", "content": self.get_confusion_matrix()}
            case OtherVisualisation.FEATURE_IMPORTANCE.name:
                return {"type": "image", "content": self.get_feature_importance()}
            case OtherVisualisation.TREE_VISUALISATION.name:
                return {"type": "image", "content": self.tree_visualisation()}
            case OtherVisualisation.DATA_DRIFT.name:
                return {"type": "image", "content": self.get_data_drift()}
            case OtherVisualisation.OUTLIER_DETECTION.name:
                return {"type": "image", "content": self.get_outlier_detection()}
            case _:
                raise ValueError("Method: " + type_visualization + " not found")

    def get_tableone(self, get_train=False):
        """_summary_

        args:

        Returns:
            _type_: _description_
        """

        data_table = self.prepared_raw_data()

        categorical = [
            "gender",
            "race",
            "first_careunit",
            "vent",
            "crrt",
            "vaso",
            "seda",
            "ami",
            "ckd",
            "copd",
            "hyperte",
            "dm",
            "sad",
            "aki",
            "stroke",
            "hosp_mort",
        ]
        groupby = "sad"

        if get_train:
            table = TableOne(
                data_table[1], categorical=categorical, groupby=groupby, pval=False
            )
        else:
            table = TableOne(
                data_table[0], categorical=categorical, groupby=groupby, pval=False
            )

        return table.tabulate(tablefmt="html")

    def get_corr_matrix(self, get_train=False, method=CorrMatrixMethod.pearson):
        """_summary_

        args:

        Returns:
            _type_: _description_
        """

        data_table = self.prepared_raw_data()

        if get_train:
            corr = (
                data_table[1].select_dtypes(exclude=["object"]).corr(method=method.name)
            )
        else:
            corr = (
                data_table[0].select_dtypes(exclude=["object"]).corr(method=method.name)
            )

        mask = np.triu(np.ones_like(corr, dtype=bool))

        f, ax = plt.subplots(figsize=(20, 16))

        cmap = sns.color_palette("viridis_r", as_cmap=True)
        hm = sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=0.3,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )

        fig = hm.get_figure()
        return self.process_image(fig)

    def get_confusion_matrix(self):

        data_table = self.prepared_raw_data()
        xgb_matrix = xgb.DMatrix(
            data_table[0].loc[:, ~data_table[0].columns.isin(["sad"])],
            label=data_table[0]["sad"],
        )

        xgb_pred_prob = self.model.predict(
            xgb_matrix
        )  # dit is voor de volledige set. vervang `xgb_matrix_full` als je de performance van het model wrt een andere set wilt.
        xgb_pred = np.where(xgb_pred_prob > 0.5, 1, 0)
        xgb_pred_factor = pd.factorize(xgb_pred)[0]
        test_sad_factor = pd.factorize(data_table[0]["sad"])[0]

        cm = metrics.confusion_matrix(xgb_pred_factor, test_sad_factor)

        accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
        recall = cm[1][1] / (cm[1][1] + cm[1][0])
        f1 = 2 * precision * recall / (precision + recall)

        title = (
            "accuracy: "
            + str(accuracy)
            + "\nprecision: "
            + str(precision)
            + "\nrecall: "
            + str(recall)
            + "\nF1-score: "
            + str(f1)
        )
        cm_display = metrics.ConfusionMatrixDisplay(
            cm, display_labels=["Non-SAD", "SAD"]
        ).plot()

        return self.process_image(cm_display)

    def get_feature_importance(self, method=ImportanceType.gain, max_features=15):
        # werkt alleen voor xgboost modellen. deze functie
        if isinstance(self.model, xgb.core.Booster):
            xgb.plot_importance(
                self.model, importance_type=method.name, max_num_features=max_features
            )
            plt.title(f"Feature importance: {method.name}")
            return self.process_image(plt)
        else:
            return False

    def tree_visualisation(self):
        # deze functie is afhankelijk van een externe programma genaamd graphviz,
        # die naast een PIP installatie ook extern een aparte installatie nodig heeft
        # https://graphviz.org/download/
        # werkt alleen voor xgboost modellen.
        if isinstance(self.model, xgb.core.Booster):
            g = xgb.to_graphviz(self.model, num_trees=self.model.best_iteration)
            g.format = "png"
            g.render(view=False)
            return self.process_image(g)
        else:
            return False

    def get_data_drift(self):
        data_table = self.prepared_raw_data()

        data_drift_report = Report(metrics=[DatasetDriftMetric(), DataDriftTable()])
        data_drift_report.run(
            reference_data=data_table[0], current_data=data_table[1]
        )
        data_drift_html = data_drift_report.get_html()
        return data_drift_report, data_drift_html

    def get_outlier_detection(self):

        data_table = self.prepared_raw_data()

        if "stay_id" in data_table[0].columns:
            data_table[0] = data_table[0].drop("stay_id", axis=1)

        data_cf = (
            data_table[0]
            .drop(
                [
                    "deliriumtime",
                    "hosp_mort",
                    "icu28dmort",
                    "icustay",
                    "hospstay",
                    "sepsistime",
                ],
                axis=1,
            )
            .dropna()
            .select_dtypes(exclude=["object"])
        )

        record = data_cf.iloc[1]
        is_categorical = [
            "race",
            "first_careunit",
            "vent",
            "ckd",
            "crrt",
            "copd",
            "gender",
            "vaso",
            "hyperte",
            "seda",
            "dm",
            "aki",
            "ami",
            "stroke",
        ]

        fig, axs = plt.subplots(6, 7, figsize=(12, 6))
        i = 0
        j = 0
        score = 0

        for c in data_cf.columns:
            axs[i, j].hist(data_cf[c], bins=20)
            title = c
            if c not in is_categorical:
                title = (
                    title
                    + "\n σ = "
                    + str(round(data_cf[c].std(), 2))
                    + "\n |x-μ| = "
                    + str(round(abs(record[c] - data_cf[c].mean()), 2))
                )
                score += abs(record[c] - data_cf[c].mean()) / data_cf[c].std()

            axs[i, j].set_title(title, fontsize=8)
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].get_yaxis().set_visible(False)
            axs[i, j].axvline(record[c], color="r", linestyle="dashed", linewidth=1)
            i += 1
            if (i % 6) == 0:
                i = 0
                j += 1

        score /= data_cf.shape[1]
        fig.suptitle("Outlier Detection", fontsize=16)
        fig.tight_layout()
        return self.process_image(fig), score

    def process_image(self, plot: plt):
        buf = BytesIO()
        plot.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        buf_content = buf.getvalue()
        buffer_size = len(buf_content)
        image_base64 = base64.b64encode(buf_content).decode("ascii")
        return image_base64

