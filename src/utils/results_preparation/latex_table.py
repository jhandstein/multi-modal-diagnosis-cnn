import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Specify the path to your Excel file and the sheet name
excel_path = "/Users/Julius/Documents/2_Uni/2_Potsdam/thesis/experiments/_quality_split/quality_split_summary.xlsx"


def main():
    # Load the sheets into DataFrames
    df_classification = pd.read_excel(excel_path, sheet_name="Classification").dropna(how="all")
    df_regression = pd.read_excel(excel_path, sheet_name="Regression").dropna(how="all")

    # Round all float columns to two decimals
    df_classification = df_classification.round(2)
    df_regression = df_regression.round(2)

    # Print the DataFrame as a LaTeX table
    print(df_classification.to_latex(index=False, float_format="%.2f"))
    print(df_regression.to_latex(index=False, float_format="%.2f"))

    # Plot grouped bars for both tasks
    # plot_grouped_bars(
    #     df_classification,
    #     task_type="Classification",
    #     ylabel="Accuracy",
    #     title="Feature map comparison for classification task",
    # )
    # plot_grouped_bars(
    #     df_regression,
    #     task_type="Regression",
    #     ylabel="MAE",
    #     title="Feature map comparison for regression task",
    # )


def plot_grouped_bars(df, task_type, title=None, ylabel=None, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    feature_maps = df.columns[1:]
    slice_planes = df[df.columns[0]]
    values = df.iloc[:, 1:].values

    num_feature_maps = len(feature_maps)
    num_slice_planes = len(slice_planes)
    bar_width = 0.15
    x = np.arange(num_feature_maps)

    # Use different colormap ranges for classification and regression
    cmap = plt.get_cmap("Set1")
    # colors = [cmap(i) for i in range(num_slice_planes)] if task_type == "Classification" else [cmap(i + 4) for i in range(num_slice_planes)]
    colors = [cmap(i) for i in range(num_slice_planes)]

    for i, slice_plane in enumerate(slice_planes):
        ax.bar(
            x + i * bar_width,
            values[i],
            width=bar_width,
            label=slice_plane,
            color=colors[i],
        )

    ax.set_ylabel(ylabel)
    ax.set_xticks(x + bar_width * (num_slice_planes - 1) / 2)
    ax.set_xticklabels(feature_maps)
    ax.legend(title="Slice Plane", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
    # plt.savefig(f"{task_type}_grouped_bars.png", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    main()
