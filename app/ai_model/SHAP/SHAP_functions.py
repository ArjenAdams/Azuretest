import base64
import enum
from io import BytesIO

import matplotlib.pyplot as plt
import shap
import xgboost as xgb


class SHAPType(enum.Enum):
    INTERACTIVE_FORCE_PLOT = 1
    BAR_PLOT = 2
    BEESWARM_PLOT = 3
    PPT_PLOT = 4
    DEPENDENCE_PLOT = 5
    LOCAL_FORCE_PLOT = 6
    LOCAL_WATERFALL_PLOT = 7
    LOCAL_DECISION_PLOT = 7
    LOCAL_MULTI_DECISION_PLOT = 8


class SHAPValues:

    """Class die alle SHAP waarden en visualisaties bevat

    Attributes:
        - tree_explainer: SHAP TreeExplainer object https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html#shap.TreeExplainer
        - explainer: SHAP Explainer object https://shap.readthedocs.io/en/latest/generated/shap.Explainer.html#shap.Explainer
        - shap_values_tree: SHAP waarden van de TreeExplainer
        - shap_values_waterfall: SHAP waarden van de Explainer
        - data: DataFrame met de data
        - model: Model object

    """

    def __init__(self, data, model):
        self.tree_explainer = -1
        self.explainer = -1
        self.shap_values_tree = -1
        self.shap_values_waterfall = -1
        self.data = data
        self.model = model

    def setup_shap_values(self) -> bool:

        """Functie die de SHAP waarden van de TreeExplainer en Explainer berekent en opslaat

        Returns:
            Bool: True als de SHAP waarden berekend zijn, anders False
        """

        if isinstance(self.model, xgb.core.Booster):
            self.tree_explainer = shap.TreeExplainer(self.model)
            self.shap_values_tree = self.tree_explainer.shap_values(
                self.data.loc[:, ~self.data.columns.isin(["sad"])]
            )
        try:
            self.explainer = shap.Explainer(self.model, self.data)
            self.shap_values_waterfall = self.explainer(self.data)

        except Exception as e:
            print(e)
            return False
        return True

    # all of these functions are depended on the model and data
    # this has to be more modular
    def generate(
        self,
        data,
        type_visualization: SHAPType,
        max_display=None,
        range=None,
        feature_name=None,
        interaction_index=None,
        index=None,
        list_indices=None,
    ) -> any:

        """Functie die de juiste SHAP visualisatie genereert op basis van meegegeven parameters

        Args:
            - data: DataFrame met de data van 1 of meerdere samples
            - type_visualization: Enum van het type SHAP visualisatie
            - max_display (Optional): Integer met het maximaal aantal features dat getoond wordt
            - range (Optional): Tuple met de range van samples die getoond worden
            - feature_name (Optional): String met de naam van de feature
            - interaction_index (Optional): Integer met de index van de interactie
            - index (Optional): Integer met de index van de sample
            - list_indices (Optional): List met de indices van de samples

        Raises:
            ValueError: Als de meegegeven SHAP visualisatie type niet bestaat

        Returns:
            SHAP object: een SHAP BytesIO object met de visualisatie encoded in base64
        """

        match type_visualization:
            case 'INTERACTIVE_FORCE_PLOT':
                return {'type': 'image', 'content': self.generate_plot_image(self.interactive_force_plot, range=range)}
            case 'BAR_PLOT':
                return {'type': 'image', 'content': self.generate_plot_image(self.bar_plot)}
            case 'BEESWARM_PLOT':
                return {'type': 'image', 'content': self.generate_plot_image(self.beeswarm_plot)}
            case 'PPT_PLOT':
                return {'type': 'image', 'content': self.generate_plot_image(self.ppt_plot, feature_name=feature_name)}
            case 'DEPENDENCE_PLOT':
                return {'type': 'image',
                        'content': self.generate_plot_image(self.dependence_plot, feature_name=feature_name,
                                                            interaction_index=interaction_index)}
            case 'LOCAL_FORCE_PLOT':
                return {'type': 'image', 'content': self.generate_plot_image(self.local_force_plot, index=index)}
            case 'LOCAL_WATERFALL_PLOT':
                return {'type': 'image', 'content': self.generate_plot_image(self.local_waterfall_plot, index=index,
                                                                             max_display=max_display)}
            case 'LOCAL_DECISION_PLOT':
                return {'type': 'image', 'content': self.generate_plot_image(self.local_decision_plot, index=index)}
            case 'LOCAL_MULTI_DECISION_PLOT':
                return {'type': 'image',
                        'content': self.generate_plot_image(self.local_multi_decision_plot, list_indices=list_indices)}
            case _:
                raise ValueError("Invalid SHAP visualization type")

    def generate_plot_image(self, plot_func, **kwargs):
        # Generate the plot
        plot_func(**kwargs)

        # Create a buffer to save the plot
        buf = BytesIO()
        plt.savefig(buf, format="png")
        # plt.savefig(buf,
        #             format="png",
        #             dpi=150,
        #             bbox_inches='tight')
        plt.close()
        # Ensure the buffer is set to the beginning before reading
        buf.seek(0)

        # Check the buffer content before encoding
        buf_content = buf.getvalue()
        buffer_size = len(buf_content)
        print(f"Buffer size after seek and getvalue: {buffer_size}")

        # Read buffer content and encode it in base64
        image_base64 = base64.b64encode(buf_content).decode("ascii")

        # Debugging: Print the length of the base64 string
        print(f"Base64 string length: {len(image_base64)}")

        return image_base64

    def generate_html(self, shap_plot):
        shap.initjs()
        return f"<head>{shap.initjs()}</head><body>{shap_plot.html()}</body>"
        # return shap_plot.html()

    def interactive_force_plot(self, range=(0, 2)):
        """Functie die een interactieve SHAP force plot genereert

        Args:
            range (tuple, optional): _description_. Defaults to (0, 2).

        Returns:
            _type_: _description_
        """

        return shap.force_plot(
            self.explainer.expected_value,
            self.shap_values_tree[range],
            self.data.loc[:, ~self.data.columns.isin(["sad"])].iloc[range],
            show=False,
        )
    #https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html
    def bar_plot(self):
        return shap.summary_plot(
            self.shap_values_tree,
            self.data.loc[:, ~self.data.columns.isin(["sad"])],
            plot_type="bar",
            show=False,
        )
    #https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html
    def beeswarm_plot(self):
        return shap.summary_plot(
            self.shap_values_tree,
            self.data.loc[:, ~self.data.columns.isin(["sad"])],
            show=False,
        )

    # https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html#Reading-SHAP-values-from-partial-dependence-plots
    def ppt_plot(self, feature_name: str):
        return shap.partial_dependence_plot(
            feature_name,
            lambda X: self.model.predict(xgb.DMatrix(X)),
            self.data,
            ice=False,
            model_expected_value=True,
            feature_expected_value=True,
            show=False,
        )

    def dependence_plot(self, feature_name: str, interaction_index=None):
        return shap.dependence_plot(
            feature_name,
            self.shap_values_tree,
            self.data.loc[:, ~self.data.columns.isin(["sad"])],
            interaction_index=interaction_index,
            show=False,
        )

    def local_force_plot(self, index):
        return shap.force_plot(
            self.explainer.expected_value,
            self.shap_values_tree[index, :],
            self.data.loc[:, ~self.data.columns.isin(["sad"])].iloc[index, :],
            show=False,
            matplotlib=True,
        )

    #https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html
    def local_waterfall_plot(self, index, max_display):
        return shap.plots.waterfall(
            self.shap_values_waterfall[index], max_display=max_display, show=False
        )

    def local_decision_plot(self, index):
        row = 0
        return shap.decision_plot(
            self.explainer.expected_value,
            self.shap_values_tree[row, :],
            self.data.loc[:, ~self.data.columns.isin(["sad"])].iloc[0, :],
            link="logit",
            highlight=0,
            show=False,
        )

    def local_multi_decision_plot(self, list_indices):
        row_current = 0

        return shap.decision_plot(
            self.explainer.expected_value,
            self.shap_values_tree[list_indices, :],
            self.data.loc[:, ~self.data.columns.isin(["sad"])].iloc[0, :],
            link="logit",
            highlight=row_current,
            show=False,
        )
