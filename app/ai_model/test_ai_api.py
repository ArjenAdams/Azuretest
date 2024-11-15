# from backend.app.ai_model.ai_api import AI_API
import unittest
from pathlib import Path

import pandas as pd

from CounterFactualType import CounterFactualType
from ai_api import AI_API


class APITestingMethods(unittest.TestCase):
    # BELANGRIJK: in verband met hoe de verschillende explainers werken, moet de data zo aangepast worden
    # Data variabele wordt gebruikt in standard API acties
    # Data_cf wordt gebruikt in counterfactuals
    mod_path = Path(__file__).parent
    data_raw = pd.read_stata((mod_path / "Modellen/MIMIC-IV.dta").resolve())
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

    def test_load_model(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
        self.assertIsNotNone(test_api.model)

    def test_predict_1(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
        test_input = self.data.iloc[0:1]
        test_input = test_input.drop("sad", axis=1)
        print(test_input)

        # gebruikt in SD_API van postman
        # test_input =  test_input.to_json(orient="records")
        test = test_api.predict(test_input)

        self.assertIsNotNone(test)
        self.assertAlmostEqual(test, 0.124599226, places=2)

    def test_predict_2(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
        test_input = self.data.iloc[5:6]
        test_input = test_input.drop("sad", axis=1)

        test = test_api.predict(test_input)

        self.assertIsNotNone(test)
        self.assertAlmostEqual(test, 0.37679648, places=2)

    def test_predict_prediction_chance(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
        test_input = self.data.iloc[0:1]
        test_input = test_input.drop("sad", axis=1)

        test = test_api.predict_probability(test_input)

        self.assertIsNotNone(test)
        # self.assertAlmostEqual(test, 0.8754008, places=2)


class CounterfactualsTestingMethods(unittest.TestCase):
    mod_path = Path(__file__).parent
    data_raw = pd.read_stata((mod_path / "Modellen/MIMIC-IV.dta").resolve())
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

    dummy_groupings = {
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

    use_feats = ["age", "weight", "temperature", "gcs"]

    def test_SetupCounterFactuals(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())

        test_api.setup_counterfactuals(
            self.data, self.dummy_groupings, self.use_feats
            )

        self.assertIsNotNone(test_api.random_cf)
        self.assertIsNotNone(test_api.genetic_cf)
        self.assertIsNotNone(test_api.KDTree_cf)

    def test_KDTreeCounterFactual(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
        test_api.setup_counterfactuals(
            self.data.drop(["sad"], axis=1), self.dummy_groupings,
            self.use_feats
        )

        test_input = self.data.iloc[0:1]
        test_input = test_input.drop("sad", axis=1)

        test = test_api.get_counterfactual(
            test_input, method=CounterFactualType.KDTREE, amount_of_cfs=2
        )

        self.assertIsNotNone(test)
        # self.assertAlmostEqual(test, 0.124599226, places=2)

        pass

    def test_GeneticCounterFactual(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
        test_api.setup_counterfactuals(
            self.data.drop(["sad"], axis=1), self.dummy_groupings,
            self.use_feats
        )
        test_input = self.data.iloc[0:1]
        test_input = test_input.drop("sad", axis=1)

        test = test_api.get_counterfactual(
            test_input, method=CounterFactualType.GENETIC, amount_of_cfs=2
        )

        self.assertIsNotNone(test)
        # self.assertAlmostEqual(test, 0.124599226, places=2)

        pass

    def test_RandomCounterFactual(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
        test_api.setup_counterfactuals(
            self.data.drop(["sad"], axis=1), self.dummy_groupings,
            self.use_feats
        )

        test_input = self.data.iloc[1:2]
        test_input = test_input.drop("sad", axis=1)

        test = test_api.get_counterfactual(
            test_input, method=CounterFactualType.RANDOM, amount_of_cfs=2
        )
        self.assertIsNotNone(test)
        # self.assertAlmostEqual(test, 0.124599226, places=2)
        pass


class SHAPVisualisationTesting(unittest.TestCase):

    #     mod_path = Path(__file__).parent
    #     data_raw = pd.read_stata((mod_path / "Modellen/MIMIC-IV.dta").resolve())
    #     data_cf = data_raw.drop(
    #         [
    #             "deliriumtime",
    #             "hosp_mort",
    #             "icu28dmort",
    #             "stay_id",
    #             "icustay",
    #             "hospstay",
    #             "sepsistime",
    #         ],
    #         axis=1,
    #     ).dropna()
    #     dummies = pd.get_dummies(data_cf["race"])
    #     data = data_cf.drop("race", axis=1).join(dummies)
    #     dummies = pd.get_dummies(data["first_careunit"])
    #     data = data.drop("first_careunit", axis=1).join(dummies)

    #     def test_setup_shap(self):
    #         test_api = AI_API()
    #         test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
    #         test_api.setup_shap_values(self.data)

    #         self.assertIsNotNone(test_api.shap_values)
    #         pass

    #     def test_interactive_force_plot(self):

    #         test_api = AI_API()
    #         test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
    #         test_api.setup_shap_values(self.data)

    #         test_input = self.data.iloc[0:1]
    #         test_input = test_input.drop("sad", axis=1)

    #         test = test_api.get_shap_visualization(
    #             test_input, SHAPType.INTERACTIVE_FORCE_PLOT, range=(0, 2)
    #         )

    #         self.assertIsNotNone(test)

    #         pass

    #     def test_bar_plot(self):

    #         test_api = AI_API()
    #         test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
    #         test_api.setup_shap_values(self.data)

    #         test_input = self.data.iloc[0:1]
    #         test_input = test_input.drop("sad", axis=1)

    #         test = test_api.get_shap_visualization(test_input, SHAPType.BAR_PLOT)

    #         self.assertIsNotNone(test)
    #         pass

    #     def test_beeswarm_plot(self):

    #         test_api = AI_API()
    #         test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
    #         test_api.setup_shap_values(self.data)

    #         test = test_api.get_shap_visualization(test_input, SHAPType.BEESWARM_PLOT)

    #         self.assertIsNotNone(test)

    #         pass

    #     def test_ppt_plot(self):

    #         test_api = AI_API()
    #         test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
    #         test_api.setup_shap_values(self.data)

    #         test_input = self.data.iloc[0:1]
    #         test = test_api.get_shap_visualization(test_input, SHAPType.PPT_PLOT, "age")

    #         self.assertIsNotNone(test)
    #         pass

    #     def test_dependence_plot(self):

    #         test_api = AI_API()
    #         test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
    #         test_api.setup_shap_values(self.data)

    #         test_input = self.data.iloc[0:1]
    #         test = test_api.get_shap_visualization(test_input, SHAPType.DEPENDENCE_PLOT)

    #         self.assertIsNotNone(test)
    #         pass

    #     def test_local_force_plot(self):

    #         test_api = AI_API()
    #         test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
    #         test_api.setup_shap_values(self.data)

    #         test_input = self.data.iloc[0:1]
    #         test_input = test_input.drop("sad", axis=1)

    #         test = test_api.get_shap_visualization(test_input, SHAPType.LOCAL_FORCE_PLOT)

    #         self.assertIsNotNone(test)

    #         pass

    #     def test_local_waterfall_plot(self):

    #         test_api = AI_API()
    #         test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
    #         test_api.setup_shap_values(self.data)

    #         test_input = self.data.iloc[0:1]
    #         test_input = test_input.drop("sad", axis=1)

    #         test = test_api.get_shap_visualization(
    #             test_input, SHAPType.LOCAL_WATERFALL_PLOT, max_display=5
    #         )

    #         self.assertIsNotNone(test)

    #         pass

    #     def test_local_decision_plot(self):

    #         test_api = AI_API()
    #         test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())
    #         test_api.setup_shap_values(self.data)

    #         test_input = self.data.iloc[0:1]
    #         test_input = test_input.drop("sad", axis=1)

    #         test = test_api.get_shap_visualization(test_input, SHAPType.LOCAL_DECISION_PLOT)

    #         self.assertIsNotNone(test)

    #         pass
    pass


class MiscVisualisation(unittest.TestCase):
    mod_path = Path(__file__).parent
    data_raw = pd.read_stata((mod_path / "Modellen/MIMIC-IV.dta").resolve())
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

    def test_tableone(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())

        test = test_api.get_tableone()

        self.assertIsNotNone(test)

    def test_correlation_matrix(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())

        test = test_api.get_corr_matrix()

        self.assertIsNotNone(test)

    def test_confusion_matrix(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())

        test = test_api.get_confusion_matrix()

        self.assertIsNotNone(test)

    def test_feature_importance(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())

        test = test_api.get_feature_importance()

        self.assertIsNotNone(test)

    # deze test faalt omdat deze functie afhankelijk van de graphviz library en het moet ergens geinstalleerd staan
    def test_tree_plot(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())

        test = test_api.tree_visualisation()

        self.assertIsNotNone(test)

    def test_data_drift(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())

        test = test_api.get_data_drift()

        self.assertIsNotNone(test)

    def test_outlier_detection(self):
        test_api = AI_API()
        test_api.load_model((self.mod_path / "Modellen/xgb.pkl").resolve())

        test = test_api.get_outlier_detection()

        self.assertIsNotNone(test)


if __name__ == "__main__":
    unittest.main()

# test_api = AI_API()
# test_api.load_model(
#     "D:/HU Projects/Dashboard-XAI-trust/backend/app/ai_model/Modellen/xgb.pkl"
# )
# test = test_api.predict(
#     "D:/HU Projects/Dashboard-XAI-trust/backend/app/ai_model/Test_inputs/test_data.json"
# )
# print(test)
#
# test_api.get_counterfactual(
#     test_api.parse_inputs(
#         "D:/HU Projects/Dashboard-XAI-trust/backend/app/ai_model/Test_inputs/test_data.json"
#     ),
#     method=1,
# )


# test = test_api.predict(
#     "D:/HU Projects/Dashboard-XAI-trust/backend/app/ai_model/Test_inputs/test1.json"
# )
# print(test)
