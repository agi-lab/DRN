import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import ticker

import numpy as np
import pandas as pd
import shap
import sklearn


class KernelSHAP_DRN:
    """
    This class produces the Kernel SHAP values regarding the distributional property of interest.
    It produces the raw Kernel SHAP values.
    It also generates SHAP dependence plot for any pair of features, considering categorical features.
    Beeswarm plot can be generated for any features.
    """

    def __init__(
        self,
        explaining_data,
        nsamples_background_fraction,
        background_data_raw: pd.DataFrame,
        value_function,
        glm_value_function,
        other_shap_values=None,
        random_state=42,
    ):
        """
        Args:
        See the DRNExplainer class for explanations regarding
        {explaining_data, nsamples_background_fraction, background_data_raw, preprocessor}
        value_function: v_{M}(S, x), given any instance x and indices S \\subseteq \\{1, ..., p\\}
        """
        self.background_data_raw = background_data_raw
        self.value_function = value_function
        self.explaining_data = explaining_data
        self.other_shap_values = other_shap_values
        self.feature_names = self.background_data_raw.columns
        sample_size = int(
            np.round(self.background_data_raw.shape[0] * nsamples_background_fraction)
        )

        if self.other_shap_values is None:
            # Compute SHAP values for the DRN network
            np.random.seed(random_state)
            kernel_shap_explainer = shap.KernelExplainer(
                self.value_function,
                shap.sample(
                    self.background_data_raw,
                    nsamples=sample_size,
                    random_state=random_state,
                ),
            )
            self.shap_values_kernel = kernel_shap_explainer(self.explaining_data)
            self.shap_base_values = self.shap_values_kernel.base_values
            self.shap_values = self.shap_values_kernel.values

            # Compute SHAP values for the GLM if required
            np.random.seed(random_state)
            self.glm_value_function = glm_value_function
            if self.glm_value_function is not None:
                kernel_shap_explainer_glm = shap.KernelExplainer(
                    self.glm_value_function,
                    shap.sample(
                        self.background_data_raw,
                        nsamples=sample_size,
                        random_state=random_state,
                    ),
                )
                self.shap_values_kernel_glm = kernel_shap_explainer_glm(
                    self.explaining_data
                )

    def forward(self):
        """
        The raw Kernel SHAP (either adjusted or DRN) output.
        """
        return self.shap_values_kernel

    def shap_glm_values(self):
        """
        The raw Kernel SHAP (GLM) output.
        """
        return self.shap_values_kernel_glm

    def shap_values_mean_adjustments(self):
        """
        The SHAP values and feature names
        """
        return (self.shap_base_values, self.shap_values, self.feature_names)

    def shap_dependence_plot(self, features_tuple, output="value"):
        """
        Create the SHAP dependence plots
        features_tuple: the pair of features required for plotting
        other_shap_values: allows for externally calculated SHAP values, i.e., FastSHAP...
        """
        tuple_indexes = [
            np.where(self.feature_names == feature)[0][0]
            for feature in features_tuple
            if feature in self.feature_names
        ]
        instances = self.explaining_data.copy()

        # Convert categorical features to numeric
        #  Encoding details
        encoders = {}
        for feature in features_tuple:
            if (
                instances[feature].dtype == object
                or instances[feature].dtype == "category"
            ):
                # not isinstance(instances[feature].values[0], float):
                encoder = sklearn.preprocessing.LabelEncoder()
                instances[feature] = encoder.fit_transform(instances[feature])
                encoders[feature] = encoder
                for class_label, encoding in zip(
                    encoder.classes_, encoder.transform(encoder.classes_)
                ):
                    print(f"{feature}: {class_label} -> {encoding}")

        if self.other_shap_values is not None:
            shap_for_tuple = self.other_shap_values
        else:
            shap_for_tuple = self.shap_value_selection(tuple_indexes, output)

        instances_for_tuple = instances.values[:, tuple_indexes]
        feature_name_tuple = self.feature_names[tuple_indexes]

        batch_size = shap_for_tuple.shape[0]

        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # 1 row, 2 columns

        for feature_idx in range(2):
            shap_values = shap_for_tuple[:, feature_idx : (feature_idx + 1)].reshape(
                batch_size, 1
            )
            feature_values = instances_for_tuple[:, feature_idx : (feature_idx + 1)]

            # Determine the color based on the other feature's value
            color_feature_index = 1 - feature_idx
            colors = instances_for_tuple[
                :, color_feature_index : (color_feature_index + 1)
            ]

            ax = axes[feature_idx]
            scatter = ax.scatter(
                feature_values, shap_values, alpha=0.85, c=colors, cmap="viridis", s=22
            )

            # Handle categorical feature for x-axis
            if feature_name_tuple[feature_idx] in encoders:
                encoder = encoders[feature_name_tuple[feature_idx]]
                ticks_and_labels = list(enumerate(encoder.classes_))
                ticks, labels = zip(*ticks_and_labels)
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels)

            # Improved Handling of Categorical Feature for Color Bar
            if feature_name_tuple[color_feature_index] in encoders:
                encoder = encoders[feature_name_tuple[color_feature_index]]
                # Create a color map with a color for each unique value in the feature
                unique_vals = np.unique(instances_for_tuple[:, color_feature_index])
                cmap = mcolors.ListedColormap(
                    plt.cm.viridis(np.linspace(0, 1, len(unique_vals)))
                )
                norm = mcolors.BoundaryNorm(
                    np.arange(-0.5, len(unique_vals) + 0.5, 1), cmap.N
                )

                scatter.set_cmap(cmap)
                scatter.set_norm(norm)

                # Create the colorbar using the specified cmap and norm
                cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(len(unique_vals)))
                cbar.set_ticklabels(encoder.classes_)
                cbar.set_label(
                    f"{feature_name_tuple[color_feature_index]}", fontsize=30
                )

                # Ensuring dot-like representation for categorical color bar
                cbar.ax.minorticks_off()  # This removes any minor ticks
                cbar.ax.get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))
            else:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(
                    f"{feature_name_tuple[color_feature_index]}", fontsize=30
                )  # Set the font size as needed

            ax.set_xlabel(f"{feature_name_tuple[feature_idx]}", fontsize=30)
            ax.set_ylabel(f"$\\phi_{{{tuple_indexes[feature_idx]+1}}}$", fontsize=30)
            ax.set_title(
                f"SHAP Values for {feature_name_tuple[feature_idx]}", fontsize=30
            )

        # plt.tick_params(axis='x', labelsize=14)
        # plt.tick_params(axis='y', labelsize=14)
        plt.tight_layout()

    def beeswarm_plot(self, features=None, output="value"):
        """
        Create the beeswarm summary plots
        features: a list of feature names required for plotting
        adjusting: False --> explaining the drn model; True --> explaining how the drn adjusts the glm
        """
        features = self.background_data_raw.columns if features is None else features
        features_idx = [
            np.where(self.feature_names == feature)[0][0]
            for feature in features
            if feature in self.feature_names
        ]

        if self.other_shap_values is not None:
            shap_values = self.other_shap_values
        else:
            shap_values = self.shap_value_selection(features_idx, output)

        # Beeswarm summary plot
        shap.summary_plot(
            shap_values, self.explaining_data.iloc[:, features_idx], plot_size=(8, 6)
        )

    def global_importance_plot(self, features=None, output="value"):
        """
        Creates a global importance plot based on the absolute SHAP values.
        """
        features = self.background_data_raw.columns if features is None else features
        features_idx = [
            np.where(self.feature_names == feature)[0][0]
            for feature in features
            if feature in self.feature_names
        ]

        if self.other_shap_values is not None:
            shap_values = self.other_shap_values
        else:
            shap_values = self.shap_value_selection(features_idx, output)

        feature_names = self.feature_names

        # Sum the absolute SHAP values for each feature across all samples
        shap_sum = np.abs(shap_values).mean(axis=0)

        # Sort the features by their importance
        feature_importance = sorted(
            zip(feature_names, shap_sum), key=lambda x: x[1], reverse=True
        )
        sorted_features, sorted_importances = zip(*feature_importance)

        # Create the plot
        plt.figure(figsize=(15, 15))
        plt.barh(
            range(len(sorted_importances)),
            sorted_importances,
            tick_label=sorted_features,
        )
        plt.xlabel("Mean Absolute SHAP Value (Global Importance)", fontsize=35)
        plt.ylabel("Features", fontsize=35)
        plt.title("Global Feature Importance Based on SHAP Values", fontsize=35)
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top

    def shap_value_selection(self, indexes, output="value"):
        if output == "value":
            shap_values = self.shap_values[:, indexes]
        elif output == "glm":
            if self.glm_value_function is None:
                raise ValueError(f"Set glm_output = True while initilising the class!")
            else:
                shap_values = self.shap_values_kernel_glm.values[:, indexes]
        elif output == "drn":
            if self.glm_value_function is None:
                raise ValueError(f"Set glm_output = True while initilising the class!")
            else:
                shap_values = (
                    self.shap_values[:, indexes]
                    + self.shap_values_kernel_glm.values[:, indexes]
                )
        return shap_values
