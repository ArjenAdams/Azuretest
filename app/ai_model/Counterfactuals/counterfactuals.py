import random

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.neighbors import KDTree


class KDTreeCounterFactual:
    """
    Essentieel hetzelfde als de KDTree van DiCE, maar dan sneller

    Instructies: initialiseer een KDTreeCounterFactual object met
    de volledige dataset die aan het dashboard gekoppeld zit,
    vervolgens kunnen naar hartelust counterfactuals gegenereerd
    worden door generate() aan te roepen op dit object.

    Attributes:
        - tree: KDTree object
        - data: DataFrame met de data, *exclusief* target
        - model: XGBoost model object
    """
    def __init__(self, data: pd.DataFrame, model: xgb.core.Booster):
        """
        :param data: data, *exclusief* target
        :param model: XGBoost model op basis waarvan we
        counterfactuals genereren
        """
        self.tree = KDTree(data)
        self.data = data
        self.model = model

    def generate(self, X: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        :param X: record waar we counterfactuals voor genereren;
        enkele rij in een pandas dataframe
        :param n: aantal counterfactuals
        :return: dataframe met counterfactuals
        """
        pred = self.model.predict(xgb.DMatrix(X))
        j = 1
        while True:
            dst, i = self.tree.query(X, k=(n * 10) ** j, return_distance=True)
            d = self.data.iloc[i[0]]
            d["reg"] = self.model.predict(xgb.DMatrix(d))
            d["pred"] = d["reg"] > 0.5
            d["dst"] = dst[0]
            if np.count_nonzero(d["pred"] == (not (pred > 0.5))) >= n:
                break
            j += 1
        return d[d["pred"] == (not (pred > 0.5))][:n]


class GeneticCounterfactual:
    """
    Vergelijkbaar met de DiCE Genetic counterfactual.

    Als je alleen in een aantal specifieke rijen wilt variëren (dus dat
    het een soort sensitivitetisanalyse wordt) kan je voor `use_feats`
    een lijst met kolomnamen meegeven, en kan je het best `limit` iets
    lager zetten. Als de class op deze manier gebruikt wordt is het
    vergelijkbaar met DiCE Random counterfactual.

    Instructies: initialiseer een GeneticCounterfactual object met:
    - de volledige dataset die aan het dashboard gekoppeld zit
    - een xgboost model
    - een dict met dummy kolommen, in het volgende format:
    {kolomnaam: [dummy-kolomnamen]}
    bij het SAD-model is dat dus:
    {
    'race':
        ['AISAN', 'BLACK', 'HISPANIC', 'OTHER', 'WHITE', 'unknown'],
    'first_careunit':
        ['CCU', 'CVICU', 'MICU', 'MICU/SICU', 'NICU', 'SICU', 'TSICU']
    }

    optioneel:
    - use_feats is een lijst met features waarin je wilt dat de
    counterfactuals variëren, als dit leeg wordt gelaten wordt er gewoon
    in iedere feature gevarieerd.
    - limit duidt aan hoe vaak het genetisch algoritme geloopt wordt,
    hoe hoger dit is hoe meer de counterfactuals dus op de input record
    lijken. Als use_feats gebruikt wordt kan je dit het best een stuk
    lager zetten, anders zit er weinig variëteit in de counterfactuals.
    - population size is (clearly) de grootte van de populatie in het
    genetisch algoritme. Dit kan je het best ongeveer even groot als de
    grootte van de dataset zetten.

    Attributes:
        - data: DataFrame met de data, *exclusief* target
        - model: XGBoost model object
        - dummy_groupings: dictionary met kolomnamen als keys en een lijst van kolomnamen als values.
        - use_feats: lijst met kolomnamen die gebruikt worden voor het genereren van counterfactuals
        - limit: integer met het maximaal aantal iteraties van het genetic algorithm
        - population_size: sample size voor het trainen van het genetic algorithm

    """

    def __init__(
        self,
        data: pd.DataFrame,
        model: xgb.core.Booster,
        dummy_groupings: dict,
        use_feats: list = None,
        limit: int = 10,
        population_size: int = 10000,
    ):
        self.dummy_groupings = dummy_groupings
        self.columns_initial = data.columns
        data = self._from_dummies(data)

        if use_feats is None:
            use_feats = data.columns
        self.data = data[use_feats]
        self.data_raw = data
        self.use_feats = use_feats
        self.use_feats_ind = data.columns.isin(use_feats)
        dummy_bool = pd.Series(self.use_feats).isin(dummy_groupings.keys())
        self.dummy_index = dummy_bool[dummy_bool == True].index
        self.lim_min = np.array(self.data.min(axis=0), dtype=np.float32)
        self.lim_max = np.array(self.data.max(axis=0), dtype=np.float32)

        self.model = model
        self.population = None
        self.population_size = population_size
        self.limit = limit

    def generate(self, X: pd.DataFrame, n: int) -> pd.DataFrame:
        """
        :param X: record waar we counterfactuals voor genereren;
        enkele rij in een pandas dataframe
        :param n: aantal counterfactuals
        :return: dataframe met counterfactuals
        """
        target = (self.model.predict(xgb.DMatrix(X)) > 0.5)[0]
        X = np.array(X)

        pred_template = np.zeros((self.population_size, self.data_raw.shape[1]))
        pred_template[:, np.where(np.invert(self.use_feats_ind))[0]] = X[0][
            np.where(np.invert(self.use_feats_ind))
        ]

        X = X[0][np.where(self.use_feats_ind)]

        self._populate()
        self._cull(X, target, pred_template)
        for i in range(self.limit):
            self._repopulate()
            self._cull(X, target, pred_template)

        fitness = np.apply_along_axis(
            lambda x: self._fitness_euclidian(X, x), 1, self.population[:n]
        )
        pred_template[: self.population.shape[0], np.where(self.use_feats_ind)[0]] = (
            self.population
        )

        res = self._to_dummies(
            pd.DataFrame(data=pred_template[:n], columns=self.data_raw.columns)
        )

        res["reg"] = self.model.predict(xgb.DMatrix(res))
        res["pred"] = res["reg"] > 0.5
        res["fitness"] = fitness
        return res

    def _populate_uniform(self) -> None:
        """
        Vult `population` met willekeurige getallen, uniform verdeeld
        over [`lim_min`, `lim_max`).

        Wordt momenteel niet gebruikt, `_populate()`
        lijkt betere resultaten te geven.
        """
        population = np.random.rand(self.population_size, self.data.shape[1])
        for i in range(population.shape[1]):
            population[:, i] = (
                population[:, i] * (self.lim_max[i] - self.lim_min[i]) + self.lim_min[i]
            )
        self.population = population

    def _populate(self) -> None:
        """
        Vult `population` met willekeurige getallen, waarbij de
        distributie van gegenereerde waarden gelijk is aan de distributie
        van de respectievelijke features.
        """
        population = (
            np.random.rand(self.population_size, self.data.shape[1])
            * self.data.shape[0]
        )
        for i in range(population.shape[1]):
            cumdist = np.sort(self.data.iloc[:, i])
            population[:, i] = cumdist[
                np.clip(population[:, i].astype(int), 0, len(cumdist) - 1)
            ]

        # TODO: ergens komen NaN's vandaan maar ik weet niet waar.
        #  dit is heel hacky er moet een beter oplossing hiervoor zijn
        for i in range(population.shape[1]):
            nan_indices = np.isnan(population[:, i])
            if np.any(nan_indices):
                valid_values = self.data.iloc[:, i].dropna().values
                population[nan_indices, i] = np.random.choice(
                    valid_values, size=nan_indices.sum()
                )

        self.population = population

    def _cull(self, X: np.ndarray, target: bool, pred_template: np.ndarray) -> None:
        """
        Verwijderd eerst alle rijen uit `population` waarvoor het
        model een voorspelling zou doen die gelijk is aan `target`,
        vervolgens wordt `population` gesorteerd en ingekort tot de
        helft van `population_size` (als dat nog niet het geval was).

        :param X: numpy array van `X` uit `generate()`
        :param target: boolean voorspelling die bij `X` hoort
        :param pred_template: numpy array vooraf gevuld met waarden die
        *niet* veranderd worden door het algoritme
        :raise: wanneer er geen counterfactuals mogelijk zijn,
        waarschijnlijk zijn er dan te weinig kolommen geselecteerd in
        `use_feats`
        """
        pred_template[:, np.where(self.use_feats_ind)[0]] = self.population
        pred = self.model.predict(
            xgb.DMatrix(
                self._to_dummies(
                    pd.DataFrame(data=pred_template, columns=self.data_raw.columns)
                )
            )
        )
        self.population = self.population[np.invert((pred > 0.5) == target)]
        self.population = np.unique(self.population, axis=0)

        if self.population.shape[0] > (self.population_size / 2):
            self._sort(X)
            self.population = self.population[: int(self.population_size / 2), :]
        if self.population.shape[0] == 0:
            raise Exception("geen mogelijke counterfactuals met deze parameters!")

    def _sort(self, X: np.ndarray) -> None:
        """
        Sorteert aan de hand van de fitness functie.

        :param X: numpy array van `X` uit `generate()`
        """
        fitness = np.apply_along_axis(
            lambda x: self._fitness_euclidian(X, x), 1, self.population
        )
        self.population = self.population[np.argsort(fitness)]

    def _repopulate(self) -> None:
        """
        Vult `population` tot `population_size` na `_cull()`, door de
        overgebleven data te recombineren.
        """
        while self.population.shape[0] <= self.population_size / 2:
            pop2 = self.population.copy()
            for i in range(pop2.shape[1]):
                if random.randint(0, 1):
                    pop2[:, i] = np.roll(pop2[:, i], 1)
            self.population = np.concatenate([self.population, pop2])

        if self.population.shape[0] < self.population_size:
            pop2 = self.population.copy()[
                : (self.population_size - self.population.shape[0])
            ]
            for i in range(pop2.shape[1]):
                if random.randint(0, 1):
                    pop2[:, i] = np.roll(pop2[:, i], 1)
            self.population = np.concatenate([self.population, pop2])

    def _fitness_euclidian(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bepaalt fitness op basis van Euclidische afstand.
        Kan wel beter maar dit is goed genoeg vooralsnog

        :param a: array waarmee we vergelijken
        :param b: array waarvan de fitness bepaald wordt
        :return: np array met respectievelijke fitness
        """
        return np.sqrt(
            np.sum(
                (np.delete(a, self.dummy_index) - np.delete(b, self.dummy_index)) ** 2
            )
        ) + sum((a[self.dummy_index] == b[self.dummy_index]).astype(float))

    def _fitness_euclidian_relative(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bepaalt fitness op basis van Euclidische afstand, relatief aan
        het bereik van de respectievelijke feature.
        Wordt niet gebruikt omdat het categoriale waarden te veel naar
        voren haalt.

        WARNING: Functie verouderd, werkt niet (naar behoren) wanneer er
        dummy-data aanwezig is.

        :param a: array waarmee we vergelijken
        :param b: array waarvan de fitness bepaald wordt
        :return: np array met respectievelijke fitness
        """
        return np.sqrt(np.sum(((np.abs(a - b) - self.lim_min) / self.lim_max) ** 2))

    def _from_dummies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Haalt dummies weg

        :param data: data waarvan dummy kolommen overeenkomen met
         `dummy_groupings`
        :return: data zonder dummies
        """
        for group in self.dummy_groupings.keys():
            d = pd.from_dummies(data[self.dummy_groupings[group]])
            d = d.replace(
                self.dummy_groupings[group], np.arange(len(self.dummy_groupings[group]))
            )
            data = data.drop(self.dummy_groupings[group], axis=1)
            data[group] = d
        return data

    def _to_dummies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Zet dummies terug

        :param data: data waarvan dummy kolommen overeenkomen met
         `dummy_groupings`
        :return: data met dummies
        """
        for group in self.dummy_groupings.keys():
            data[group] = data[group].replace(
                np.arange(len(self.dummy_groupings[group])), self.dummy_groupings[group]
            )
            d = pd.get_dummies(data[group])
            data = data.drop(group, axis=1).join(d)
            if len(self.dummy_groupings[group]) != len(d.columns):
                diff = np.setdiff1d(self.dummy_groupings[group], d.columns)
                data[diff] = pd.DataFrame(
                    np.zeros(((data.shape[0]), len(diff))), dtype=bool
                )
        data = data[self.columns_initial]
        return data
