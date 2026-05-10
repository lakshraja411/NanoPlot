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
    line_width,
    show_markers,
    marker_size,
    show_legend,
    x_min,
    x_max,
    y_min,
    y_max,
    custom_xticks,
    custom_yticks,
    figure_width,
    figure_height,
    dpi,
):
    """Create an Origin-style matplotlib figure."""

    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=dpi)

    for col in y_cols:
        y_data = convert_current(df[col], current_unit)

        if show_markers:
            ax.plot(
                df[x_col],
                y_data,
                linewidth=line_width,
                marker="o",
                markersize=marker_size,
                label=col,
            )
        else:
            ax.plot(
                df[x_col],
                y_data,
                linewidth=line_width,
                label=col,
            )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    if title.strip():
        ax.set_title(title, fontsize=13)

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
        direction="in",
        length=4,
        width=1,
        labelsize=10,
    )

    ax.minorticks_off()

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    if show_legend:
        ax.legend(frameon=False, fontsize=10)

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
st.sidebar.subheader("Style")

line_width = st.sidebar.slider("Line width", 0.5, 5.0, 2.0, 0.1)
show_markers = st.sidebar.checkbox("Show markers", value=False)
marker_size = st.sidebar.slider("Marker size", 1.0, 10.0, 4.0, 0.5)
show_legend = st.sidebar.checkbox("Show legend", value=True)

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
        line_width=line_width,
        show_markers=show_markers,
        marker_size=marker_size,
        show_legend=show_legend,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        custom_xticks=custom_xticks,
        custom_yticks=custom_yticks,
        figure_width=figure_width,
        figure_height=figure_height,
        dpi=dpi,
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
