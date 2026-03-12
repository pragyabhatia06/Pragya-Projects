import math
from datetime import datetime
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="GIS Data Engineering Dashboard",
    page_icon="🌍",
    layout="wide",
)


# -----------------------------------
# Safe display helpers
# -----------------------------------
def safe_str(value):
    if pd.isna(value):
        return ""
    return str(value)


def make_display_safe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(safe_str)
    return out


def render_html_table(df: pd.DataFrame, height: int = 420):
    if df.empty:
        st.info("No data available.")
        return

    safe_df = make_display_safe(df)

    html = safe_df.to_html(index=False, escape=False)
    styled_html = f"""
    <div style="
        max-height:{height}px;
        overflow:auto;
        border:1px solid #ddd;
        border-radius:8px;
        padding:8px;
        background-color:white;
    ">
        {html}
    </div>
    """

    st.markdown(
        """
        <style>
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        th, td {
            border: 1px solid #e6e6e6;
            padding: 8px;
            text-align: left;
            vertical-align: top;
        }
        th {
            background-color: #f7f7f7;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(styled_html, unsafe_allow_html=True)


# -----------------------------------
# Sample maritime GIS dataset
# -----------------------------------
@st.cache_data
def load_sample_port_data() -> pd.DataFrame:
    csv_text = """port_name,country,region,lat,lon,port_type,annual_calls
London Gateway,United Kingdom,Europe,51.5074,0.0015,Container,1600
Felixstowe,United Kingdom,Europe,51.9630,1.3510,Container,3200
Southampton,United Kingdom,Europe,50.9097,-1.4044,Cruise,2100
Liverpool,United Kingdom,Europe,53.4084,-3.0189,Container,1400
Rotterdam,Netherlands,Europe,51.9475,4.1427,Container,7200
Antwerp,Belgium,Europe,51.2637,4.4003,Container,5400
Hamburg,Germany,Europe,53.5461,9.9661,Container,5000
Le Havre,France,Europe,49.4938,0.1079,Container,1800
Valencia,Spain,Europe,39.4489,-0.3162,Container,4100
Genoa,Italy,Europe,44.4056,8.9463,Cruise,2300
Piraeus,Greece,Europe,37.9420,23.6465,Cruise,2500
Singapore,Singapore,Asia,1.2644,103.8405,Container,8100
Dubai,UAE,Middle East,25.2697,55.3088,Container,4600
New York,USA,North America,40.6840,-74.0062,Container,3900
Los Angeles,USA,North America,33.7361,-118.2631,Container,6100
"""
    return pd.read_csv(StringIO(csv_text))


# -----------------------------------
# Geospatial helpers
# -----------------------------------
def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["lat_valid"] = out["lat"].between(-90, 90)
    out["lon_valid"] = out["lon"].between(-180, 180)
    out["coordinate_status"] = out.apply(
        lambda row: "VALID" if row["lat_valid"] and row["lon_valid"] else "INVALID",
        axis=1,
    )

    return out


def haversine_nm(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in nautical miles.
    """
    r_km = 6371.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = r_km * c
    distance_nm = distance_km * 0.539957

    return round(distance_nm, 2)


def build_route_table(df: pd.DataFrame, origin_ports: list, destination_ports: list) -> pd.DataFrame:
    rows = []

    origin_df = df[df["port_name"].isin(origin_ports)]
    destination_df = df[df["port_name"].isin(destination_ports)]

    for _, o in origin_df.iterrows():
        for _, d in destination_df.iterrows():
            if o["port_name"] == d["port_name"]:
                continue

            distance_nm = haversine_nm(o["lat"], o["lon"], d["lat"], d["lon"])

            rows.append(
                {
                    "origin_port": o["port_name"],
                    "destination_port": d["port_name"],
                    "origin_country": o["country"],
                    "destination_country": d["country"],
                    "distance_nm": distance_nm,
                    "origin_type": o["port_type"],
                    "destination_type": d["port_type"],
                }
            )

    route_df = pd.DataFrame(rows)
    if not route_df.empty:
        route_df = route_df.sort_values("distance_nm").reset_index(drop=True)

    return route_df


def build_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        rows.append(
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isna().sum()),
                "non_null_count": int(df[col].notna().sum()),
                "unique_count": int(df[col].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------
# Load + transform
# -----------------------------------
@st.cache_data
def prepare_port_data() -> pd.DataFrame:
    df = load_sample_port_data()
    df.columns = [str(col).strip() for col in df.columns]

    df["port_name"] = df["port_name"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df["region"] = df["region"].astype(str).str.strip()
    df["port_type"] = df["port_type"].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["annual_calls"] = pd.to_numeric(df["annual_calls"], errors="coerce").fillna(0).astype(int)

    df = validate_coordinates(df)
    return df


# -----------------------------------
# Charts
# -----------------------------------
def render_calls_by_region(df: pd.DataFrame):
    if df.empty:
        st.info("No chart data available.")
        return

    region_df = (
        df.groupby("region")["annual_calls"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(region_df.index.astype(str), region_df.values)
    ax.set_title("Annual Calls by Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Annual Calls")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


def render_port_type_mix(df: pd.DataFrame):
    if df.empty:
        st.info("No chart data available.")
        return

    type_df = df["port_type"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(type_df.index.astype(str), type_df.values)
    ax.set_title("Port Type Distribution")
    ax.set_xlabel("Port Type")
    ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)


# -----------------------------------
# UI
# -----------------------------------
st.title("🌍 GIS Data Engineering Dashboard")
st.caption(
    "A geospatial data engineering demo for port analytics, coordinate validation, and route distance analysis."
)

st.markdown(
    """
### What this project demonstrates
- geospatial data ingestion
- coordinate validation
- transformation into analytics-ready GIS format
- route distance calculation in nautical miles
- data quality reporting
- GIS dashboard presentation
"""
)

port_df = prepare_port_data()

with st.sidebar:
    st.header("Filters")

    selected_regions = st.multiselect(
        "Region",
        sorted(port_df["region"].dropna().unique().tolist()),
        default=sorted(port_df["region"].dropna().unique().tolist()),
    )

    selected_port_types = st.multiselect(
        "Port Type",
        sorted(port_df["port_type"].dropna().unique().tolist()),
        default=sorted(port_df["port_type"].dropna().unique().tolist()),
    )

filtered_df = port_df[
    port_df["region"].isin(selected_regions)
    & port_df["port_type"].isin(selected_port_types)
].copy()

if filtered_df.empty:
    st.warning("No ports available for the selected filters.")
    st.stop()

# KPIs
total_ports = len(filtered_df)
valid_coords = int((filtered_df["coordinate_status"] == "VALID").sum())
invalid_coords = int((filtered_df["coordinate_status"] == "INVALID").sum())
total_calls = int(filtered_df["annual_calls"].sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Ports", total_ports)
k2.metric("Valid Coordinates", valid_coords)
k3.metric("Invalid Coordinates", invalid_coords)
k4.metric("Annual Calls", f"{total_calls:,}")

st.markdown("---")

# Map
st.subheader("Port Map")

fig = px.scatter_geo(
    filtered_df,
    lat="lat",
    lon="lon",
    hover_name="port_name",
    hover_data={
        "country": True,
        "region": True,
        "port_type": True,
        "annual_calls": True,
        "lat": False,
        "lon": False,
    },
    size="annual_calls",
    projection="natural earth",
    title="Port Locations",
)

fig.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Port Dataset Preview")
    render_html_table(
        filtered_df[
            [
                "port_name",
                "country",
                "region",
                "lat",
                "lon",
                "port_type",
                "annual_calls",
                "coordinate_status",
            ]
        ],
        height=320,
    )

with right:
    st.subheader("Data Quality Report")
    dq_df = build_data_quality_report(filtered_df)
    render_html_table(dq_df, height=320)

st.markdown("---")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Annual Calls by Region")
    render_calls_by_region(filtered_df)

with c2:
    st.subheader("Port Type Mix")
    render_port_type_mix(filtered_df)

st.markdown("---")

st.subheader("Route Distance Calculator")

origin_ports = st.multiselect(
    "Select Origin Ports",
    filtered_df["port_name"].tolist(),
    default=filtered_df["port_name"].tolist()[:2],
    key="origin_ports",
)

destination_ports = st.multiselect(
    "Select Destination Ports",
    filtered_df["port_name"].tolist(),
    default=filtered_df["port_name"].tolist()[2:5] if len(filtered_df) >= 5 else filtered_df["port_name"].tolist(),
    key="destination_ports",
)

route_df = build_route_table(filtered_df, origin_ports, destination_ports)

if route_df.empty:
    st.info("Select origin and destination ports to calculate route distances.")
else:
    route_kpi_1, route_kpi_2 = st.columns(2)
    route_kpi_1.metric("Calculated Routes", len(route_df))
    route_kpi_2.metric("Shortest Route (NM)", f"{route_df['distance_nm'].min():,.2f}")

    render_html_table(route_df, height=320)

st.markdown("---")

with st.expander("Engineering Notes"):
    st.markdown(
        f"""
**Source layer**
- Sample maritime port reference dataset loaded into pandas

**Transformation layer**
- Standardized columns
- Converted latitude/longitude to numeric
- Validated coordinate ranges
- Prepared route combinations for analysis

**GIS analytics layer**
- Port map visualization
- Distance calculation using haversine formula
- Route table in nautical miles
- Region and port-type aggregation

**Data quality layer**
- Coordinate validation flags
- Per-column profiling metrics
- Filtered dataset inspection

**Business relevance**
- Suitable for port planning, marine logistics, berth analytics, and route scenario modelling
"""
    )