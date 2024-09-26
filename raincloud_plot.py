import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def raincloud_plot(
    data: pd.DataFrame,
    features: list,
    x_label: str,
    title: str,
    show_boxes: bool = True,
    x_log_scale: bool = False,
) -> None:
    """Create a raincloud plot for a dataset.

    Args:
        data (pd.DataFrame): The dataset to plot.
        features (list): The features to plot.
        x_label (str): The label for the x-axis.
        title (str): The title of the plot.
        show_boxes (bool): Whether to show boxplots. Defaults to True.
        x_log_scale (bool): Whether to use a log scale for the x-axis. Defaults to False.
    """
    number_of_features = len(features)
    data_selection = data[features].select_dtypes(include=[np.number])
    colours = list(mcolors.TABLEAU_COLORS.values())
    rain_offset = 0.15

    if number_of_features > len(colours):
        colours = colours * (number_of_features // len(colours) + 1)

    # Create figure and axis
    _, ax = plt.subplots(figsize=(8, number_of_features * 2))

    # Boxplot data
    if show_boxes:
        rain_offset = 0
        bp = ax.boxplot(
            x=data_selection,
            widths=0.1,
            patch_artist=True,
            vert=False,
            positions=[x + 0.9 for x in range(number_of_features)],
            showfliers=False,
        )

        # Change the color of the boxes
        _ = [patch.set_facecolor(color) for patch, color in zip(bp["boxes"], colours)]
        _ = [patch.set_alpha(0.8) for patch in bp["boxes"]]
        _ = [median.set_color("black") for median in bp["medians"]]

    # Violinplot data
    vp = ax.violinplot(
        data_selection,
        points=500,
        showmeans=False,
        showextrema=False,
        showmedians=False,
        vert=False,
    )

    # Modify violin plot so we only see the upper half
    for idx, b in enumerate(vp["bodies"]):
        b.get_paths()[0].vertices[:, 1] = np.clip(
            b.get_paths()[0].vertices[:, 1], idx + 1, idx + 2
        )
        b.set_color(colours[idx])
        b.set_alpha(0.8)
        b.set_edgecolor("black")

    # Scatterplot data
    for idx, column in enumerate(features):
        y = np.full(len(data_selection[column]), idx + 0.75 + rain_offset).astype(float)
        idxs = np.arange(len(y))
        y.flat[idxs] += np.random.uniform(low=-0.05, high=0.05, size=len(idxs))
        plt.scatter(data_selection[column], y, s=0.5, c=colours[idx])

    if x_log_scale:
        ax.set_xscale("log")

    plt.yticks(np.arange(1, number_of_features + 1, 1), features)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()
