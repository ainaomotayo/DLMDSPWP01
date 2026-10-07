'''
Module visualizer
Implements interactive data visualization using the Bokeh library.
Follows Unit 3.4.3 of the course book.
'''

import os
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, Band


def resolve_output_path(filepath):
    '''
    Resolves output file path relative to project root if needed.
    '''
    if not filepath:
        return "outputs/visualization.html"
    if os.path.isabs(filepath):
        return filepath
    if os.path.exists(os.path.dirname(filepath)) and os.path.dirname(filepath):
        return filepath

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, filepath)


class DataVisualizer(object):
    '''
    Generates interactive Bokeh visual representations of training data,
    chosen ideal functions, test cases, and deviation bounds.
    '''

    def __init__(self, output_filepath="outputs/visualization.html"):
        '''
        Constructor for DataVisualizer.
        output_filepath: destination file path for generated HTML visualization.
        '''
        self.output_filepath = resolve_output_path(output_filepath)
        parent_dir = os.path.dirname(self.output_filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    def generate_visualization(self, train_df, ideal_df, chosen_models, all_test_summary):
        '''
        Compiles interactive multi-panel Bokeh visualization.
        train_df: training DataFrame.
        ideal_df: ideal functions DataFrame.
        chosen_models: dictionary containing chosen ideal function regression data.
        all_test_summary: list of dictionaries containing all test points and mapping status.
        return: output HTML filepath.
        '''
        output_file(self.output_filepath, title="DLMDSPWP01 Function Fitting and Mapping Analysis")

        plots = []
        function_pairs = [("y1", "Function 1"), ("y2", "Function 2"),
                          ("y3", "Function 3"), ("y4", "Function 4")]

        color_palette = {
            "y1": {"train": "#1f77b4", "ideal": "#ff7f0e", "mapped": "#2ca02c"},
            "y2": {"train": "#9467bd", "ideal": "#8c564b", "mapped": "#d62728"},
            "y3": {"train": "#e377c2", "ideal": "#7f7f7f", "mapped": "#bcbd22"},
            "y4": {"train": "#17becf", "ideal": "#aec7e8", "mapped": "#ff9896"}
        }

        # Build 4 subplots, one for each chosen training and ideal pair
        for t_col, label in function_pairs:
            model_info = chosen_models[t_col]
            i_col = model_info["chosen_ideal_col"]
            thresh = model_info["threshold"]

            plot_title = (f"{label}: Training {t_col} vs Chosen Ideal {i_col} "
                          f"(Max Dev: {model_info['max_deviation']:.4f}, Threshold: {thresh:.4f})")

            p = figure(
                title=plot_title,
                x_axis_label="x",
                y_axis_label="y",
                width=800,
                height=350,
                tools="pan,box_zoom,wheel_zoom,reset,save"
            )

            # Ideal curve
            ideal_source = ColumnDataSource(data={
                "x": ideal_df["x"],
                "y": ideal_df[i_col],
                "upper": ideal_df[i_col] + thresh,
                "lower": ideal_df[i_col] - thresh
            })

            # Tolerance band representation of the deviation
            band = Band(
                base="x",
                lower="lower",
                upper="upper",
                source=ideal_source,
                level="underlay",
                fill_alpha=0.15,
                fill_color=color_palette[t_col]["ideal"],
                line_width=1,
                line_color=color_palette[t_col]["ideal"]
            )
            p.add_layout(band)

            # Ideal function line
            p.line(
                "x", "y",
                source=ideal_source,
                line_width=2.5,
                color=color_palette[t_col]["ideal"],
                legend_label=f"Ideal {i_col}"
            )

            # Training data points
            train_source = ColumnDataSource(data={
                "x": train_df["x"],
                "y": train_df[t_col]
            })
            p.scatter(
                "x", "y",
                source=train_source,
                marker="circle",
                size=4,
                color=color_palette[t_col]["train"],
                alpha=0.6,
                legend_label=f"Training {t_col}"
            )

            # Mapped test points for this specific ideal function
            mapped_for_this = [
                pt for pt in all_test_summary
                if pt["mapped"] and pt["ideal_func"].upper() == i_col.upper()
            ]

            if mapped_for_this:
                test_source = ColumnDataSource(data={
                    "x": [pt["x"] for pt in mapped_for_this],
                    "y": [pt["y"] for pt in mapped_for_this],
                    "delta_y": [pt["delta_y"] for pt in mapped_for_this]
                })

                test_renderer = p.scatter(
                    "x", "y",
                    source=test_source,
                    marker="diamond",
                    size=10,
                    color=color_palette[t_col]["mapped"],
                    legend_label=f"Assigned Test ({len(mapped_for_this)} pts)"
                )

                hover = HoverTool(
                    renderers=[test_renderer],
                    tooltips=[
                        ("X", "@x{0.00}"),
                        ("Y", "@y{0.0000}"),
                        ("Deviation (Delta Y)", "@delta_y{0.0000}")
                    ]
                )
                p.add_tools(hover)

            p.legend.location = "top_left"
            p.legend.click_policy = "hide"
            plots.append(p)

        # Overview plot showing test data mapping status
        summary_plot = figure(
            title="Overview: Test Dataset Point Classification (Mapped vs Unmapped)",
            x_axis_label="x",
            y_axis_label="y",
            width=800,
            height=400,
            tools="pan,box_zoom,wheel_zoom,reset,save"
        )

        mapped_pts = [pt for pt in all_test_summary if pt["mapped"]]
        unmapped_pts = [pt for pt in all_test_summary if not pt["mapped"]]

        if mapped_pts:
            m_source = ColumnDataSource(data={
                "x": [pt["x"] for pt in mapped_pts],
                "y": [pt["y"] for pt in mapped_pts],
                "delta": [pt["delta_y"] for pt in mapped_pts],
                "ideal": [pt["ideal_func"] for pt in mapped_pts]
            })
            r_mapped = summary_plot.scatter(
                "x", "y",
                source=m_source,
                marker="circle",
                size=8,
                color="#2ca02c",
                alpha=0.8,
                legend_label=f"Mapped Test Cases ({len(mapped_pts)})"
            )
            summary_plot.add_tools(HoverTool(
                renderers=[r_mapped],
                tooltips=[
                    ("X", "@x{0.00}"),
                    ("Y", "@y{0.0000}"),
                    ("Assigned Ideal", "@ideal"),
                    ("Delta Y", "@delta{0.0000}")
                ]
            ))

        if unmapped_pts:
            u_source = ColumnDataSource(data={
                "x": [pt["x"] for pt in unmapped_pts],
                "y": [pt["y"] for pt in unmapped_pts]
            })
            summary_plot.scatter(
                "x", "y",
                source=u_source,
                marker="cross",
                size=7,
                color="#7f7f7f",
                alpha=0.5,
                legend_label=f"Unmapped Test Cases ({len(unmapped_pts)})"
            )

        summary_plot.legend.location = "top_left"
        summary_plot.legend.click_policy = "hide"
        plots.append(summary_plot)

        layout = column(*plots)
        save(layout)
        return self.output_filepath

    def export_png(self, train_df, ideal_df, chosen_models, all_test_summary,
                   output_dir="outputs/figures", combined_filename="visualization_combined.png"):
        '''
        Exports high-resolution static PNG figures using Matplotlib (Unit 3.4.1).
        Generates individual chart files for insertion into academic report documents,
        as well as a combined multi-panel dashboard.

        train_df: training DataFrame.
        ideal_df: ideal functions DataFrame.
        chosen_models: dictionary of chosen ideal regression models.
        all_test_summary: list of test classification records.
        output_dir: directory path to store figure PNG files.
        combined_filename: filename for the stacked composite figure.
        return: dictionary containing filepaths for individual and combined figures.
        '''
        import matplotlib.pyplot as plt

        resolved_dir = resolve_output_path(output_dir)
        os.makedirs(resolved_dir, exist_ok=True)

        function_keys = [("y1", "Function 1"), ("y2", "Function 2"),
                         ("y3", "Function 3"), ("y4", "Function 4")]

        palette = {
            "y1": {"train": "#1f77b4", "ideal": "#ff7f0e", "mapped": "#2ca02c"},
            "y2": {"train": "#9467bd", "ideal": "#8c564b", "mapped": "#d62728"},
            "y3": {"train": "#e377c2", "ideal": "#7f7f7f", "mapped": "#bcbd22"},
            "y4": {"train": "#17becf", "ideal": "#aec7e8", "mapped": "#ff9896"}
        }

        generated_files = {"individual": [], "combined": None}

        # 1. Generate individual figures (one per function for report embedding)
        for idx, (t_col, label) in enumerate(function_keys):
            info = chosen_models[t_col]
            i_col = info["chosen_ideal_col"]
            thresh = info["threshold"]

            fig, ax = plt.subplots(figsize=(9, 5))

            # Ideal curve
            ax.plot(
                ideal_df["x"],
                ideal_df[i_col],
                color=palette[t_col]["ideal"],
                linewidth=2.0,
                label=f"Chosen Ideal {i_col.upper()}"
            )

            # Tolerance band
            ax.fill_between(
                ideal_df["x"],
                ideal_df[i_col] - thresh,
                ideal_df[i_col] + thresh,
                color=palette[t_col]["ideal"],
                alpha=0.2,
                label=f"Tolerance Band (+/- {thresh:.4f})"
            )

            # Training data points
            ax.scatter(
                train_df["x"],
                train_df[t_col],
                color=palette[t_col]["train"],
                s=14,
                alpha=0.6,
                label=f"Training {t_col}"
            )

            # Assigned test data points
            mapped_pts = [
                pt for pt in all_test_summary
                if pt["mapped"] and pt["ideal_func"].upper() == i_col.upper()
            ]

            if mapped_pts:
                ax.scatter(
                    [pt["x"] for pt in mapped_pts],
                    [pt["y"] for pt in mapped_pts],
                    color=palette[t_col]["mapped"],
                    marker="D",
                    s=45,
                    edgecolors="black",
                    linewidth=0.8,
                    label=f"Assigned Test ({len(mapped_pts)} pts)"
                )

            title = (f"Figure {idx + 1}: Training {t_col} vs Ideal {i_col.upper()} "
                     f"(Max Dev: {info['max_deviation']:.4f}, Threshold: {thresh:.4f})")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("x", fontsize=10)
            ax.set_ylabel("y", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left", fontsize=8)

            single_path = os.path.join(resolved_dir, f"figure_{idx + 1}_training_{t_col}_vs_ideal_{i_col}.png")
            plt.tight_layout()
            plt.savefig(single_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            generated_files["individual"].append(single_path)

        # Overview figure for test point classification
        fig_summary, ax_summary = plt.subplots(figsize=(9, 5))
        mapped_all = [pt for pt in all_test_summary if pt["mapped"]]
        unmapped_all = [pt for pt in all_test_summary if not pt["mapped"]]

        if mapped_all:
            ax_summary.scatter(
                [pt["x"] for pt in mapped_all],
                [pt["y"] for pt in mapped_all],
                color="#2ca02c",
                marker="o",
                s=40,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.5,
                label=f"Mapped Test Cases ({len(mapped_all)} pts)"
            )

        if unmapped_all:
            ax_summary.scatter(
                [pt["x"] for pt in unmapped_all],
                [pt["y"] for pt in unmapped_all],
                color="#7f7f7f",
                marker="x",
                s=35,
                alpha=0.6,
                label=f"Unmapped Test Cases ({len(unmapped_all)} pts)"
            )

        ax_summary.set_title("Figure 5: Test Dataset Classification (Mapped vs Unmapped)",
                             fontsize=11, fontweight="bold")
        ax_summary.set_xlabel("x", fontsize=10)
        ax_summary.set_ylabel("y", fontsize=10)
        ax_summary.grid(True, linestyle=":", alpha=0.6)
        ax_summary.legend(loc="upper left", fontsize=8)

        overview_path = os.path.join(resolved_dir, "figure_5_test_data_classification_overview.png")
        plt.tight_layout()
        plt.savefig(overview_path, dpi=300, bbox_inches="tight")
        plt.close(fig_summary)
        generated_files["individual"].append(overview_path)

        # 2. Combined multi-panel figure
        fig_combined, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 22))
        for idx, (t_col, label) in enumerate(function_keys):
            ax = axes[idx]
            info = chosen_models[t_col]
            i_col = info["chosen_ideal_col"]
            thresh = info["threshold"]

            ax.plot(ideal_df["x"], ideal_df[i_col], color=palette[t_col]["ideal"],
                    linewidth=2.0, label=f"Chosen Ideal {i_col.upper()}")
            ax.fill_between(ideal_df["x"], ideal_df[i_col] - thresh, ideal_df[i_col] + thresh,
                            color=palette[t_col]["ideal"], alpha=0.2, label=f"Tolerance Band (+/- {thresh:.4f})")
            ax.scatter(train_df["x"], train_df[t_col], color=palette[t_col]["train"],
                       s=12, alpha=0.6, label=f"Training {t_col}")

            mapped_pts = [pt for pt in all_test_summary if pt["mapped"] and pt["ideal_func"].upper() == i_col.upper()]
            if mapped_pts:
                ax.scatter([pt["x"] for pt in mapped_pts], [pt["y"] for pt in mapped_pts],
                           color=palette[t_col]["mapped"], marker="D", s=40, edgecolors="black",
                           linewidth=0.8, label=f"Assigned Test ({len(mapped_pts)} pts)")

            title = (f"{label}: Training {t_col} vs Ideal {i_col.upper()} "
                     f"(Max Dev: {info['max_deviation']:.4f}, Threshold: {thresh:.4f})")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("x", fontsize=10)
            ax.set_ylabel("y", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left", fontsize=8)

        ax_comb_sum = axes[4]
        if mapped_all:
            ax_comb_sum.scatter([pt["x"] for pt in mapped_all], [pt["y"] for pt in mapped_all],
                                color="#2ca02c", marker="o", s=35, alpha=0.85, edgecolors="black",
                                linewidth=0.5, label=f"Mapped Test Cases ({len(mapped_all)} pts)")
        if unmapped_all:
            ax_comb_sum.scatter([pt["x"] for pt in unmapped_all], [pt["y"] for pt in unmapped_all],
                                color="#7f7f7f", marker="x", s=30, alpha=0.6,
                                label=f"Unmapped Test Cases ({len(unmapped_all)} pts)")

        ax_comb_sum.set_title("Overview: Test Dataset Classification (Mapped vs Unmapped)",
                              fontsize=11, fontweight="bold")
        ax_comb_sum.set_xlabel("x", fontsize=10)
        ax_comb_sum.set_ylabel("y", fontsize=10)
        ax_comb_sum.grid(True, linestyle=":", alpha=0.6)
        ax_comb_sum.legend(loc="upper left", fontsize=8)

        comb_path = os.path.join(resolved_dir, combined_filename)
        plt.tight_layout()
        plt.savefig(comb_path, dpi=300, bbox_inches="tight")
        plt.close(fig_combined)
        generated_files["combined"] = comb_path

        return generated_files
