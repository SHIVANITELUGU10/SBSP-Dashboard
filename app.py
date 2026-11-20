import streamlit as st
import pandas as pd
import numpy as np
import math
import io
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="SBSP Simulation Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SBSP MODEL
# ============================================================

C = 299_792_458.0  # m/s

def safe_db_to_linear(db):
    return 10.0 ** (db / 10.0)

def calculate(params):
    """
    Simplified end-to-end SBSP model for dashboard demonstration.

    Assumption:
    The single Antenna Gain input is used for both transmitting and
    receiving antenna gain. Replace this with separate Gt/Gr values
    when your physical model provides them.

    The dashboard deliberately keeps the equations visible and simple.
    For your final project, the same function can be replaced by the
    exact Octave/MATLAB model.
    """
    G = safe_db_to_linear(params["antenna_gain_db"])
    f = params["frequency_ghz"] * 1e9
    d = params["distance_km"] * 1e3
    wavelength = C / f

    solar_input = params["irradiance"] * params["pv_area"]
    pv_output = solar_input * params["pv_efficiency"] / 100.0

    mppt_factor = 1.0 if params["mppt"] else 0.92
    dc_dc_output = pv_output * params["dc_dc_efficiency"] / 100.0 * mppt_factor

    # RF power available after DC-DC stage and transmitter efficiency.
    rf_available = dc_dc_output * params["tx_efficiency"] / 100.0
    rf_transmitted = min(params["tx_power"], rf_available)

    # Friis transmission equation.
    path_factor = (wavelength / (4.0 * math.pi * d)) ** 2
    received_rf = rf_transmitted * G * G * path_factor

    # Rectenna conversion.
    rectified_dc = received_rf * params["rectenna_efficiency"] / 100.0

    total_efficiency = (
        100.0 * rectified_dc / solar_input if solar_input > 0 else 0.0
    )

    fspl_db = 20.0 * math.log10(4.0 * math.pi * d / wavelength)

    # Simple dashboard-level ripple indicator.
    # This is NOT a transistor-level rectifier ripple model.
    stage_factor = max(0.25, 1.0 - 0.04 * max(params["rectifier_stages"] - 1, 0))
    filter_factor = 0.35 if params["ripple_filter"] else 1.0
    ripple_percent = (
        100.0
        * (1.0 - params["rectenna_efficiency"] / 100.0)
        * stage_factor
        * filter_factor
    )

    return {
        "Solar Input Power (W)": solar_input,
        "PV Output Power (W)": pv_output,
        "DC-DC Output Power (W)": dc_dc_output,
        "RF Transmitted Power (W)": rf_transmitted,
        "Received RF Power (W)": received_rf,
        "Rectified DC Power (W)": rectified_dc,
        "Total System Efficiency (%)": total_efficiency,
        "Free-Space Path Loss (dB)": fspl_db,
        "Output Ripple Indicator (%)": ripple_percent,
        "Wavelength (m)": wavelength,
        "MPPT Factor": mppt_factor,
    }

def distance_sweep(params, start_km=100, stop_km=3000, step_km=100):
    distances = np.arange(start_km, stop_km + step_km, step_km)
    rows = []
    for d in distances:
        p = params.copy()
        p["distance_km"] = float(d)
        r = calculate(p)
        rows.append({
            "Distance (km)": d,
            "Received RF Power (W)": r["Received RF Power (W)"],
            "Rectified DC Power (W)": r["Rectified DC Power (W)"],
            "Path Loss (dB)": r["Free-Space Path Loss (dB)"],
            "Total Efficiency (%)": r["Total System Efficiency (%)"],
        })
    return pd.DataFrame(rows)

def pv_efficiency_sweep(params):
    values = np.arange(5, 51, 5)
    rows = []
    for v in values:
        p = params.copy()
        p["pv_efficiency"] = float(v)
        r = calculate(p)
        rows.append({
            "PV Efficiency (%)": v,
            "PV Output Power (W)": r["PV Output Power (W)"],
            "Rectified DC Power (W)": r["Rectified DC Power (W)"],
            "Total Efficiency (%)": r["Total System Efficiency (%)"],
        })
    return pd.DataFrame(rows)

def antenna_gain_sweep(params):
    values = np.arange(0, 41, 2)
    rows = []
    for v in values:
        p = params.copy()
        p["antenna_gain_db"] = float(v)
        r = calculate(p)
        rows.append({
            "Antenna Gain (dBi)": v,
            "Received RF Power (W)": r["Received RF Power (W)"],
            "Rectified DC Power (W)": r["Rectified DC Power (W)"],
        })
    return pd.DataFrame(rows)

def frequency_sweep(params):
    values = np.arange(1, 11, 0.5)
    rows = []
    for v in values:
        p = params.copy()
        p["frequency_ghz"] = float(v)
        r = calculate(p)
        rows.append({
            "Frequency (GHz)": v,
            "Path Loss (dB)": r["Free-Space Path Loss (dB)"],
            "Received RF Power (W)": r["Received RF Power (W)"],
        })
    return pd.DataFrame(rows)

def monte_carlo(params, n=1000, seed=42):
    rng = np.random.default_rng(seed)

    irr = rng.normal(params["irradiance"], params["irradiance"] * 0.05, n)
    pv_eff = rng.normal(params["pv_efficiency"], 1.5, n)
    tx_eff = rng.normal(params["tx_efficiency"], 1.0, n)
    rect_eff = rng.normal(params["rectenna_efficiency"], 2.0, n)
    distance = rng.normal(params["distance_km"], params["distance_km"] * 0.03, n)

    irr = np.clip(irr, 50, None)
    pv_eff = np.clip(pv_eff, 1, 100)
    tx_eff = np.clip(tx_eff, 1, 100)
    rect_eff = np.clip(rect_eff, 1, 100)
    distance = np.clip(distance, 1, None)

    rows = []
    for i in range(n):
        p = params.copy()
        p["irradiance"] = float(irr[i])
        p["pv_efficiency"] = float(pv_eff[i])
        p["tx_efficiency"] = float(tx_eff[i])
        p["rectenna_efficiency"] = float(rect_eff[i])
        p["distance_km"] = float(distance[i])
        r = calculate(p)
        rows.append({
            "Irradiance": irr[i],
            "PV Efficiency": pv_eff[i],
            "TX Efficiency": tx_eff[i],
            "Rectenna Efficiency": rect_eff[i],
            "Distance (km)": distance[i],
            "Rectified DC Power (W)": r["Rectified DC Power (W)"],
            "Total Efficiency (%)": r["Total System Efficiency (%)"]
        })
    return pd.DataFrame(rows)

def sensitivity_analysis(params):
    base = calculate(params)["Rectified DC Power (W)"]
    tests = {
        "Solar Irradiance": ("irradiance", 0.10),
        "PV Area": ("pv_area", 0.10),
        "PV Efficiency": ("pv_efficiency", 0.10),
        "DC-DC Efficiency": ("dc_dc_efficiency", 0.10),
        "TX Efficiency": ("tx_efficiency", 0.10),
        "TX Power": ("tx_power", 0.10),
        "Antenna Gain": ("antenna_gain_db", 1.0),
        "Distance": ("distance_km", 0.10),
        "Rectenna Efficiency": ("rectenna_efficiency", 0.10),
        "Load Resistance": ("load_resistance", 0.10),
    }

    rows = []
    for name, (key, change) in tests.items():
        p = params.copy()
        if key == "antenna_gain_db":
            p[key] = params[key] + change
        else:
            p[key] = params[key] * (1 + change)

        new_value = calculate(p)["Rectified DC Power (W)"]
        sensitivity = (
            ((new_value - base) / base) * 100.0 if abs(base) > 1e-15 else 0.0
        )
        rows.append({
            "Parameter": name,
            "Change": "+10%" if key != "antenna_gain_db" else "+1 dB",
            "Output Change (%)": sensitivity
        })

    return pd.DataFrame(rows).sort_values("Output Change (%)", ascending=False)

def make_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("☀️ SBSP Inputs")
st.sidebar.caption("Change parameters and press Run Simulation.")

st.sidebar.subheader("Solar / PV")

irradiance = st.sidebar.number_input(
    "Solar Irradiance (W/m²)", min_value=0.0, value=1000.0, step=50.0
)
pv_area = st.sidebar.number_input(
    "PV Area (m²)", min_value=0.01, value=10.0, step=1.0
)
pv_efficiency = st.sidebar.slider(
    "PV Efficiency (%)", 1.0, 50.0, 20.0, 0.5
)

st.sidebar.subheader("Power Conversion")

dc_dc_efficiency = st.sidebar.slider(
    "DC-DC Converter Efficiency (%)", 1.0, 100.0, 95.0, 0.5
)
tx_efficiency = st.sidebar.slider(
    "Transmitter Efficiency (%)", 1.0, 100.0, 90.0, 0.5
)
tx_power = st.sidebar.number_input(
    "Transmitter Power (W)", min_value=0.0, value=1700.0, step=50.0
)

st.sidebar.subheader("RF Link")

frequency_ghz = st.sidebar.number_input(
    "Frequency (GHz)", min_value=0.01, value=2.45, step=0.05
)
antenna_gain_db = st.sidebar.number_input(
    "Antenna Gain (dBi)", min_value=0.0, value=30.0, step=1.0
)
distance_km = st.sidebar.number_input(
    "Transmission Distance (km)", min_value=1.0, value=1000.0, step=50.0
)

st.sidebar.subheader("Rectenna / Load")

rectenna_efficiency = st.sidebar.slider(
    "Rectenna Efficiency (%)", 1.0, 100.0, 70.0, 0.5
)
load_resistance = st.sidebar.number_input(
    "Load Resistance (Ω)", min_value=0.1, value=50.0, step=5.0
)

st.sidebar.subheader("Configuration")

mppt = st.sidebar.checkbox("MPPT Enable", value=True)
ripple_filter = st.sidebar.checkbox("Ripple Filter Enable", value=True)
rectifier_stages = st.sidebar.number_input(
    "Rectifier Stages (CMOS)", min_value=1, max_value=20, value=4, step=1
)
pvt_corner = st.sidebar.selectbox(
    "PVT Corner", ["TT", "FF", "SS", "FS", "SF"]
)

run = st.sidebar.button("▶ RUN SIMULATION", use_container_width=True)

params = {
    "irradiance": irradiance,
    "pv_area": pv_area,
    "pv_efficiency": pv_efficiency,
    "dc_dc_efficiency": dc_dc_efficiency,
    "tx_efficiency": tx_efficiency,
    "tx_power": tx_power,
    "frequency_ghz": frequency_ghz,
    "antenna_gain_db": antenna_gain_db,
    "distance_km": distance_km,
    "rectenna_efficiency": rectenna_efficiency,
    "load_resistance": load_resistance,
    "mppt": mppt,
    "ripple_filter": ripple_filter,
    "rectifier_stages": rectifier_stages,
    "pvt_corner": pvt_corner,
}

# ============================================================
# MAIN
# ============================================================

st.title("☀️ Satellite Based Solar Power System")
st.subheader("Simulation Dashboard & Data Visualization")

st.info(
    "This dashboard is a software visualization layer for the SBSP model. "
    "The default model uses the Friis transmission equation and simple stage efficiencies. "
    "Replace the calculation function with your validated Octave/MATLAB model for final project results."
)

if run or "results" not in st.session_state:
    st.session_state["results"] = calculate(params)

results = st.session_state["results"]

# ---------------- KPI CARDS ----------------

st.markdown("### Key Performance Indicators")

c1, c2, c3 = st.columns(3)
c1.metric("Solar Input Power", f"{results['Solar Input Power (W)']:.2f} W")
c2.metric("PV Output Power", f"{results['PV Output Power (W)']:.2f} W")
c3.metric("RF Transmitted Power", f"{results['RF Transmitted Power (W)']:.2f} W")

c4, c5, c6 = st.columns(3)
c4.metric("Received RF Power", f"{results['Received RF Power (W)']:.6g} W")
c5.metric("Rectified DC Power", f"{results['Rectified DC Power (W)']:.6g} W")
c6.metric("Total System Efficiency", f"{results['Total System Efficiency (%)']:.4f} %")

# ---------------- SECONDARY RESULTS ----------------

st.markdown("### RF / Rectenna Results")

r1, r2, r3, r4 = st.columns(4)
r1.metric("Free-Space Path Loss", f"{results['Free-Space Path Loss (dB)']:.2f} dB")
r2.metric("Wavelength", f"{results['Wavelength (m)']:.4g} m")
r3.metric("Output Ripple Indicator", f"{results['Output Ripple Indicator (%)']:.3f} %")
r4.metric("PVT Corner", pvt_corner)

# ---------------- TABS ----------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Main Simulation",
    "📈 Parameter Sweep",
    "🎲 Monte Carlo",
    " sensitivity Analysis",
    "📋 Data"
])

with tab1:
    st.markdown("### Energy Flow Through the System")

    energy = pd.DataFrame({
        "Stage": [
            "Solar Input",
            "PV Output",
            "DC-DC Output",
            "RF Transmitted",
            "Received RF",
            "Rectified DC"
        ],
        "Power (W)": [
            results["Solar Input Power (W)"],
            results["PV Output Power (W)"],
            results["DC-DC Output Power (W)"],
            results["RF Transmitted Power (W)"],
            results["Received RF Power (W)"],
            results["Rectified DC Power (W)"]
        ]
    })

    fig = px.bar(
        energy,
        x="Stage",
        y="Power (W)",
        title="Power at Different System Stages",
        text_auto=".3s"
    )
    fig.update_layout(xaxis_title="", yaxis_title="Power (W)")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        loss_values = [
            max(results["Solar Input Power (W)"] - results["PV Output Power (W)"], 0),
            max(results["PV Output Power (W)"] - results["DC-DC Output Power (W)"], 0),
            max(results["DC-DC Output Power (W)"] - results["RF Transmitted Power (W)"], 0),
            max(results["RF Transmitted Power (W)"] - results["Received RF Power (W)"], 0),
            max(results["Received RF Power (W)"] - results["Rectified DC Power (W)"], 0),
        ]
        loss_df = pd.DataFrame({
            "Stage Loss": [
                "PV Loss",
                "DC-DC / MPPT Loss",
                "Transmitter Loss",
                "Free-Space / Link Loss",
                "Rectenna Loss"
            ],
            "Loss (W)": loss_values
        })
        fig_loss = px.bar(
            loss_df,
            x="Stage Loss",
            y="Loss (W)",
            title="Estimated Energy Loss at Each Stage",
            text_auto=".3s"
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_b:
        config_df = pd.DataFrame({
            "Parameter": [
                "Solar Irradiance",
                "PV Area",
                "PV Efficiency",
                "DC-DC Efficiency",
                "TX Efficiency",
                "TX Power",
                "Frequency",
                "Antenna Gain",
                "Distance",
                "Rectenna Efficiency",
                "Load Resistance",
                "MPPT",
                "Ripple Filter",
                "CMOS Rectifier Stages",
                "PVT Corner"
            ],
            "Value": [
                f"{irradiance:.2f} W/m²",
                f"{pv_area:.2f} m²",
                f"{pv_efficiency:.2f} %",
                f"{dc_dc_efficiency:.2f} %",
                f"{tx_efficiency:.2f} %",
                f"{tx_power:.2f} W",
                f"{frequency_ghz:.3f} GHz",
                f"{antenna_gain_db:.2f} dBi",
                f"{distance_km:.2f} km",
                f"{rectenna_efficiency:.2f} %",
                f"{load_resistance:.2f} Ω",
                "Enabled" if mppt else "Disabled",
                "Enabled" if ripple_filter else "Disabled",
                str(rectifier_stages),
                pvt_corner
            ]
        })
        st.dataframe(config_df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Parameter Sweep")

    sweep_type = st.selectbox(
        "Select sweep",
        [
            "Transmission Distance",
            "PV Efficiency",
            "Antenna Gain",
            "Frequency"
        ]
    )

    if sweep_type == "Transmission Distance":
        df = distance_sweep(params)
        fig = px.line(
            df,
            x="Distance (km)",
            y=["Received RF Power (W)", "Rectified DC Power (W)"],
            markers=True,
            title="Power vs Transmission Distance"
        )
    elif sweep_type == "PV Efficiency":
        df = pv_efficiency_sweep(params)
        fig = px.line(
            df,
            x="PV Efficiency (%)",
            y=["PV Output Power (W)", "Rectified DC Power (W)"],
            markers=True,
            title="Output Power vs PV Efficiency"
        )
    elif sweep_type == "Antenna Gain":
        df = antenna_gain_sweep(params)
        fig = px.line(
            df,
            x="Antenna Gain (dBi)",
            y=["Received RF Power (W)", "Rectified DC Power (W)"],
            markers=True,
            title="Received Power vs Antenna Gain"
        )
    else:
        df = frequency_sweep(params)
        fig = px.line(
            df,
            x="Frequency (GHz)",
            y=["Path Loss (dB)", "Received RF Power (W)"],
            markers=True,
            title="Frequency vs Path Loss / Received Power"
        )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Sweep CSV",
        data=make_csv(df),
        file_name="ParameterSweep.csv",
        mime="text/csv"
    )

with tab3:
    st.markdown("### Monte Carlo Analysis")
    n = st.slider("Number of Monte Carlo samples", 100, 5000, 1000, 100)
    mc = monte_carlo(params, n=n)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(
        "Mean Rectified DC",
        f"{mc['Rectified DC Power (W)'].mean():.6g} W"
    )
    mc2.metric(
        "Std. Deviation",
        f"{mc['Rectified DC Power (W)'].std():.6g} W"
    )
    mc3.metric(
        "Maximum",
        f"{mc['Rectified DC Power (W)'].max():.6g} W"
    )

    fig_mc = px.histogram(
        mc,
        x="Rectified DC Power (W)",
        nbins=40,
        title="Monte Carlo Distribution of Rectified DC Power"
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    st.dataframe(mc.head(100), use_container_width=True, hide_index=True)

    st.download_button(
        "Download Monte Carlo CSV",
        data=make_csv(mc),
        file_name="MonteCarloResults.csv",
        mime="text/csv"
    )

with tab4:
    st.markdown("### Sensitivity Analysis")
    sens = sensitivity_analysis(params)

    fig_s = px.bar(
        sens,
        x="Parameter",
        y="Output Change (%)",
        title="Sensitivity of Rectified DC Power",
        text_auto=".2f"
    )
    fig_s.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_s, use_container_width=True)

    st.dataframe(sens, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Sensitivity CSV",
        data=make_csv(sens),
        file_name="SensitivityResults.csv",
        mime="text/csv"
    )

with tab5:
    st.markdown("### Current Simulation Data")

    output_df = pd.DataFrame({
        "Output Parameter": list(results.keys()),
        "Value": list(results.values())
    })

    st.dataframe(output_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Simulation Results",
        data=make_csv(output_df),
        file_name="SimulationResults.csv",
        mime="text/csv"
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "SBSP Simulation Dashboard | Team 9 | "
    "Streamlit + Python + Octave/MATLAB compatible workflow"
)
