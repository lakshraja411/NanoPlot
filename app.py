import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="OriginLite Plotter",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_data(uploaded_file):
    """Load CSV or Excel file into a pandas DataFrame."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        # comment='#' helps with instrument-exported CSV files that contain metadata
        return pd.read_csv(uploaded_file, comment="#").dropna(how="all")

    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file).dropna(how="all")

    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


def convert_current(series, unit):
    """Convert current from ampere to chosen display unit."""
    factors = {
        "A": 1,
        "mA": 1e3,
        "µA": 1e6,
        "nA": 1e9,
        "pA": 1e12,
    }
    return series * factors[unit]


def make_plot(
    df,
    x_col,
    y_cols,
    current_unit,
    x_label,
    y_label,
    title,
    show_legend,
    legend_fontsize,
    legend_location,
    use_custom_legend_position,
    legend_x,
    legend_y,
    legend_box,
    axis_label_fontsize,
    tick_fontsize,
    tick_direction,
    x_min,
    x_max,
    y_min,
    y_max,
    custom_xticks,
    custom_yticks,
    figure_width,
    figure_height,
    dpi,
    curve_styles,
    show_origin_lines,
    origin_line_color,
    origin_line_width,
):
    """Create a clean Origin-style matplotlib figure."""

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.linewidth": 1.4,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.labelcolor": "black",
        "legend.frameon": False,
    })

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=dpi)

    for col in y_cols:
        y_data = convert_current(df[col], current_unit)
        style = curve_styles[col]

        if style["show_markers"]:
            ax.plot(
                df[x_col],
                y_data,
                color=style["color"],
                linestyle=style["line_style"],
                linewidth=style["line_width"],
                marker=style["marker"],
                markersize=style["marker_size"],
                markerfacecolor=style["marker_face_color"],
                markeredgecolor=style["marker_edge_color"],
                markeredgewidth=style["marker_edge_width"],
                label=style["label"],
            )
        else:
            ax.plot(
                df[x_col],
                y_data,
                color=style["color"],
                linestyle=style["line_style"],
                linewidth=style["line_width"],
                label=style["label"],
            )

    ax.set_xlabel(x_label, fontsize=axis_label_fontsize, labelpad=8)
    ax.set_ylabel(y_label, fontsize=axis_label_fontsize, labelpad=8)

    if title.strip():
        ax.set_title(title, fontsize=axis_label_fontsize, pad=10)

    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    if custom_xticks.strip():
        ticks = [float(x.strip()) for x in custom_xticks.split(",") if x.strip()]
        ax.set_xticks(ticks)

    if custom_yticks.strip():
        ticks = [float(y.strip()) for y in custom_yticks.split(",") if y.strip()]
        ax.set_yticks(ticks)

    ax.tick_params(
        axis="both",
        which="major",
        direction=tick_direction,
        length=5,
        width=1.2,
        labelsize=tick_fontsize,
        top=False,
        right=False,
    )

    ax.minorticks_off()
    ax.grid(False)

    if show_origin_lines:
        ax.axhline(
            y=0,
            color=origin_line_color,
            linestyle="--",
            linewidth=origin_line_width,
            zorder=0,
        )

        ax.axvline(
            x=0,
            color=origin_line_color,
            linestyle="--",
            linewidth=origin_line_width,
            zorder=0,
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        spine.set_color("black")

    if show_legend:
        if use_custom_legend_position:
            legend = ax.legend(
                frameon=legend_box,
                fontsize=legend_fontsize,
                loc="center",
                bbox_to_anchor=(legend_x, legend_y),
                handlelength=2.5,
                edgecolor="black",
                facecolor="white",
                framealpha=1.0,
            )
        else:
            legend = ax.legend(
                frameon=legend_box,
                fontsize=legend_fontsize,
                loc=legend_location,
                handlelength=2.5,
                edgecolor="black",
                facecolor="white",
                framealpha=1.0,
            )

    fig.tight_layout()
    return fig


def fig_to_bytes(fig, file_format="png", dpi=300):
    """Convert matplotlib figure to downloadable bytes."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format=file_format, dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    return buffer


# ============================================================
# MAIN APP
# ============================================================
st.title("📈 OriginLite Plotter")
st.caption("A clean Origin-style plotting tool for IV curves and nanopore data.")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to start plotting.")
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.subheader("Data preview")
st.dataframe(df.head(), use_container_width=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least two numeric columns: one X column and one Y column.")
    st.stop()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Plot controls")

x_col = st.sidebar.selectbox(
    "X column",
    numeric_cols,
    index=0,
)

default_y_cols = numeric_cols[1:]

y_cols = st.sidebar.multiselect(
    "Y columns",
    numeric_cols,
    default=default_y_cols,
)

if not y_cols:
    st.warning("Select at least one Y column.")
    st.stop()

current_unit = st.sidebar.selectbox(
    "Display current unit",
    ["A", "mA", "µA", "nA", "pA"],
    index=3,
)

x_label = st.sidebar.text_input("X-axis label", value="Voltage (V)")
y_label = st.sidebar.text_input("Y-axis label", value=f"Current ({current_unit})")
title = st.sidebar.text_input("Plot title", value="")

st.sidebar.divider()
st.sidebar.subheader("Global style")

axis_label_fontsize = st.sidebar.slider("Axis label font size", 8, 24, 13, 1)
tick_fontsize = st.sidebar.slider("Tick font size", 6, 20, 11, 1)
tick_direction = st.sidebar.selectbox(
    "Tick direction",
    ["out", "in", "inout"],
    index=0,
)

show_legend = st.sidebar.checkbox("Show legend", value=True)
legend_fontsize = st.sidebar.slider("Legend font size", 6, 20, 10, 1)
legend_box = st.sidebar.checkbox("Add box around legend", value=True)

legend_location = st.sidebar.selectbox(
    "Legend preset position",
    [
        "best",
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "center right",
        "center left",
        "lower center",
        "upper center",
        "center",
    ],
    index=0,
)

use_custom_legend_position = st.sidebar.checkbox(
    "Use custom legend position",
    value=False,
)

legend_x = st.sidebar.slider(
    "Legend X position",
    -0.5,
    1.5,
    0.75,
    0.01,
)

legend_y = st.sidebar.slider(
    "Legend Y position",
    -0.5,
    1.5,
    0.85,
    0.01,
)

st.sidebar.divider()
st.sidebar.subheader("Curve styles")

origin_colors = [
    "#000000",  # black
    "#FF0000",  # red
    "#0000FF",  # blue
    "#008000",  # green
    "#FF00FF",  # magenta
    "#FFA500",  # orange
    "#800080",  # purple
    "#8B4513",  # brown
]
line_styles = {
    "Solid": "-",
    "Dashed": "--",
    "Dotted": ":",
    "Dash-dot": "-.",
}
marker_styles = {
    "Circle": "o",
    "Square": "s",
    "Triangle": "^",
    "Diamond": "D",
    "Cross": "x",
    "Plus": "+",
}

curve_styles = {}

for i, col in enumerate(y_cols):
    with st.sidebar.expander(f"Style: {col}", expanded=(i < 2)):
        label = st.text_input(
            f"Legend label for {col}",
            value=str(col),
            key=f"label_{col}",
        )
        color = st.color_picker(
            f"Line colour for {col}",
            value=origin_colors[i % len(origin_colors)],
            key=f"color_{col}",
        )
        line_width = st.slider(
            f"Line width for {col}",
            0.5,
            6.0,
            2.0,
            0.1,
            key=f"lw_{col}",
        )
        line_style_name = st.selectbox(
            f"Line style for {col}",
            list(line_styles.keys()),
            index=0,
            key=f"ls_{col}",
        )
        show_markers = st.checkbox(
            f"Show markers for {col}",
            value=False,
            key=f"markers_{col}",
        )
        marker_name = st.selectbox(
            f"Marker shape for {col}",
            list(marker_styles.keys()),
            index=1,
            key=f"marker_shape_{col}",
        )
        marker_size = st.slider(
            f"Marker size for {col}",
            1.0,
            12.0,
            4.0,
            0.5,
            key=f"ms_{col}",
        )
        marker_face_color = st.color_picker(
            f"Marker fill colour for {col}",
            value=color,
            key=f"mfc_{col}",
        )
        marker_edge_color = st.color_picker(
            f"Marker edge colour for {col}",
            value="#000000",
            key=f"mec_{col}",
        )
        marker_edge_width = st.slider(
            f"Marker edge width for {col}",
            0.0,
            3.0,
            0.5,
            0.1,
            key=f"mew_{col}",
        )

        curve_styles[col] = {
            "label": label,
            "color": color,
            "line_width": line_width,
            "line_style": line_styles[line_style_name],
            "show_markers": show_markers,
            "marker": marker_styles[marker_name],
            "marker_size": marker_size,
            "marker_face_color": marker_face_color,
            "marker_edge_color": marker_edge_color,
            "marker_edge_width": marker_edge_width,
        }

st.sidebar.divider()
st.sidebar.subheader("Reference lines")

show_origin_lines = st.sidebar.checkbox(
    "Show dashed origin lines",
    value=False,
)

origin_line_color = st.sidebar.color_picker(
    "Origin line colour",
    value="#808080",
)

origin_line_width = st.sidebar.slider(
    "Origin line width",
    0.5,
    3.0,
    1.0,
    0.1,
)

st.sidebar.divider()
st.sidebar.subheader("Axis limits")

use_x_limits = st.sidebar.checkbox("Set X limits", value=False)
if use_x_limits:
    x_min = st.sidebar.number_input("X min", value=float(df[x_col].min()))
    x_max = st.sidebar.number_input("X max", value=float(df[x_col].max()))
else:
    x_min = None
    x_max = None

# Estimate y range after conversion
all_y_converted = pd.concat([convert_current(df[col], current_unit) for col in y_cols])

use_y_limits = st.sidebar.checkbox("Set Y limits", value=False)
if use_y_limits:
    y_min = st.sidebar.number_input("Y min", value=float(all_y_converted.min()))
    y_max = st.sidebar.number_input("Y max", value=float(all_y_converted.max()))
else:
    y_min = None
    y_max = None

st.sidebar.divider()
st.sidebar.subheader("Custom ticks")

custom_xticks = st.sidebar.text_input(
    "X ticks, comma-separated",
    value="-0.6,-0.4,-0.2,0,0.2,0.4,0.6",
)

custom_yticks = st.sidebar.text_input(
    "Y ticks, comma-separated",
    value="",
)

st.sidebar.divider()
st.sidebar.subheader("Export settings")

figure_width = st.sidebar.number_input("Figure width", value=5.0, min_value=2.0, max_value=12.0, step=0.5)
figure_height = st.sidebar.number_input("Figure height", value=4.0, min_value=2.0, max_value=12.0, step=0.5)
dpi = st.sidebar.selectbox("DPI", [150, 300, 600], index=1)

# ============================================================
# MAKE PLOT
# ============================================================
try:
    fig = make_plot(
        df=df,
        x_col=x_col,
        y_cols=y_cols,
        current_unit=current_unit,
        x_label=x_label,
        y_label=y_label,
        title=title,
        show_legend=show_legend,
        legend_fontsize=legend_fontsize,
        legend_location=legend_location,
        use_custom_legend_position=use_custom_legend_position,
        legend_x=legend_x,
        legend_y=legend_y,
        legend_box=legend_box,
        axis_label_fontsize=axis_label_fontsize,
        tick_fontsize=tick_fontsize,
        tick_direction=tick_direction,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        custom_xticks=custom_xticks,
        custom_yticks=custom_yticks,
        figure_width=figure_width,
        figure_height=figure_height,
        dpi=dpi,
        curve_styles=curve_styles,
        show_origin_lines=show_origin_lines,
        origin_line_color=origin_line_color,
        origin_line_width=origin_line_width,
    )
except Exception as e:
    st.error(f"Could not create plot: {e}")
    st.stop()

st.subheader("Plot")
st.pyplot(fig, use_container_width=False)

# ============================================================
# DOWNLOAD BUTTONS
# ============================================================
png_bytes = fig_to_bytes(fig, "png", dpi=dpi)
pdf_bytes = fig_to_bytes(fig, "pdf", dpi=dpi)
svg_bytes = fig_to_bytes(fig, "svg", dpi=dpi)

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        "Download PNG",
        data=png_bytes,
        file_name="originlite_plot.png",
        mime="image/png",
    )

with col2:
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="originlite_plot.pdf",
        mime="application/pdf",
    )

with col3:
    st.download_button(
        "Download SVG",
        data=svg_bytes,
        file_name="originlite_plot.svg",
        mime="image/svg+xml",
    )

st.success("Plot generated successfully.")
