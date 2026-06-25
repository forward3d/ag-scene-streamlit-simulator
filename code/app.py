# -*- coding: utf-8 -*-
"""
Created on Mon Nov 03 00:36:44 2026

@author: AshnerGeraldNovilla
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import time
import itertools
import tempfile
import os
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64

# --- Import our new modules ---
from session_manager import init_session
from data_loader import load_client_dataset, load_client_model, get_meridian_resource

# For Multi Processing
import pickle
from concurrent.futures import as_completed
import warnings

# --- Warnings Filter ---
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, message="A `organic_media` value was passed in the `new_data` argument.*")
warnings.filterwarnings("ignore", category=UserWarning, message="A `media_spend` value was passed in the `new_data` argument.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="arviz")
warnings.filterwarnings("ignore", category=UserWarning, message="The `aggregate_geos` argument is ignored in the national model.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="ROI, mROI, Effectiveness, and CPIK are not reported*")

# --- Global Variables ---
date_column = "time"
geo_column = "geo"
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent

logo_img_path = BASE_DIR / "images" / "assembly_icon.png"
logo_path = str(BASE_DIR / "images" / "logo.svg")

if logo_img_path.is_file():
    logo_img = Image.open(logo_img_path)
else:
    logo_img = None 

# --- UI Helper Functions ---
def image_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def geo_changed():
    st.session_state.opt_media_result  = None
    st.session_state.opt_budget_bounds = None
    st.session_state.opt_media_summary = None
    st.session_state.opt_paid_media_vars = None
    st.session_state.total_result_future = None
    st.session_state.forecast_func = None

def collapse_forecast_expander():
    if 'forecast_expander_state' in st.session_state:
        st.session_state.forecast_expander_state = False

def selector_filter(uploaded_file, file_path):
    # Process the uploaded file
    if uploaded_file is not None:    
        data = pd.read_csv(uploaded_file)
    else:
        data = file_path

    # This is important to be provided
    key_mapping = {
        'Channel0_spend': 'Channel0',
        'Channel1_spend': 'Channel1',
        'Channel2_spend': 'Channel2',
        'Channel3_spend': 'Channel3',
        'Channel4_spend': 'Channel4',
    }
    
    rename_map = {old_name: new_name for new_name, old_name in key_mapping.items()}
    data = data.rename(columns=rename_map)
    data_col_sort = np.array(data[sorted(data.columns)].columns)
    
    default_paid_media = list(key_mapping.keys())
    default_non_paid_media = [col for col in data_col_sort if "visitors_" in col]

    st.subheader("Modeling Selections and Configuration")    
    col1, col2, col3 = st.columns(3)

    with col1:
        geo_option = st.multiselect("Geo Selection", tuple(data[geo_column].unique()), max_selections=1 , key="geo_option",default=data[geo_column].unique()[0] ,on_change=geo_changed)
        if len(data[geo_column].unique())<=1:
            geo_option = 'national_geo'

    with col2:
        date_start = st.date_input("Start Date", pd.to_datetime('2023-10-30'))

    with col3:
        date_end = st.date_input("End Date", pd.to_datetime(data[date_column]).max(), max_value=pd.to_datetime(data[date_column]).max()) 
        
    col3, col4, col5 = st.columns(3)
    with col3:
        paid_media_vars = st.multiselect("Paid Media Variable Selections", data_col_sort, default=default_paid_media)
        paid_media_vars = [key_mapping[item] for item in paid_media_vars]
    with col4:
        non_paid_media_vars = st.multiselect("Non Paid Media Variable Selections", data_col_sort, default=default_non_paid_media)
    with col5:
        control_vars = st.multiselect("Control Media Variable Selections", data_col_sort, default=None)
        
    col6, col7, col8 = st.columns(3)
    with col6:
        y_col = st.multiselect("Target Variable", data[sorted(data.columns)].columns, max_selections=1,  key="target", default=['conversions'])
    with col7:
        spend_mul_inc = st.number_input("Spend Multiplier Increments", min_value=1.0, max_value=9.0, step=1e-1, format="%.2f", value=3.20)
    with col8:
        n_reference = st.number_input("Nth Referencer", value=1)
        
    col9, col10, col11 = st.columns(3)
    with col9:
        total_budget = st.number_input("Total Budget", value=None, placeholder="Enter Total Budget...")
    with col10:
        maxiter = st.number_input("Maximum Iteration", value=500)
    with col11:
        popsize = st.number_input("Batch Size", value=32)

    def max_divisor(n, max_val):
        for i in range(max_val, 0, -1):
            if n % i == 0:
                return i
        return 1

    num_cols = max_divisor(len(paid_media_vars), 6)
    rows = [paid_media_vars[i:i + num_cols] for i in range(0, len(paid_media_vars), num_cols)]

    st.subheader("Budget Bounds Input")
    with st.expander("🔧 Adjust Budget Bounds Settings", expanded=False):
        st.write("You can customize the maximum and minimum spend limits for each media channel here. Adjusting these values will affect the optimization outcomes.")
        inputs = {}
        st.markdown(
            """
            <style>
              .stCaption p {
                font-size: 5px !important;
                margin-bottom: 1px !important;
                margin-top: 1px !important;
                line-height: 1.0 !important;
              }
            </style>
            """, unsafe_allow_html=True
        )
        for row in rows:
            cols = st.columns(len(row))
            for col, name in zip(cols, row):
                with col:
                    st.markdown(f"<p style='font-size:9px; margin:0 0 2px 0; line-height:1'>{name} (%)</p>", unsafe_allow_html=True)
                    pct = st.number_input('', min_value=0, max_value=100, step=1, value=50, format="%d", label_visibility="hidden", key=name)
                    inputs[name] = pct / 100.0

    st.subheader("Dataset Preview")
    dt = data.copy()
    dt[date_column] = pd.to_datetime(dt[date_column])
    start_date_compare = pd.to_datetime(date_start)
    end_date_compare = pd.to_datetime(date_end)
    dt = data.loc[(data[geo_column].isin(geo_option)) & (dt[date_column]>= start_date_compare) & (dt[date_column]<= end_date_compare)]
    
    with st.expander("🔼🔽 Dataset Preview", expanded=False):
        st.dataframe(pd.concat([dt.head(), dt.tail()], ignore_index=True), use_container_width=True, hide_index=True)

    return dt, date_column, geo_option, y_col, paid_media_vars, non_paid_media_vars, control_vars, date_start, date_end, spend_mul_inc, n_reference, inputs, total_budget, maxiter, popsize

# --- Plotting Functions ---
def waterfall_plot(df):
    driver_df = df
    driver_df.loc[len(driver_df)] = ["Total", '']
    baseline_data = driver_df.set_index('Driver').copy()
    baseline_data = baseline_data.loc['Baseline', 'Total Contribution']
    measure = ['absolute'] + ['relative'] * (len(driver_df) - 2) + ['total']
    
    fig = go.Figure(go.Waterfall(
        name="Driver Impact", orientation="v", measure=measure,
        x=driver_df["Driver"], y=driver_df["Total Contribution"], base=baseline_data,
        decreasing={"marker":{"color":"Maroon", "line":{"color":"red", "width":2}}},
        increasing={"marker":{"color":"Teal"}},
        totals={"marker":{"color":"orange", "line":{"color":"orange", "width":3}}},
        connector={"line": {"color": "gray"}}
    ))
    fig.update_layout(title="Waterfall Chart of Media Contributions", xaxis_title="Media", yaxis_title="Total Contribution", showlegend=False, height=600, margin=dict(l=20, r=20, t=80, b=20))
    return fig

def pie_plot(df):
    driver_df = df.loc[df['Driver']!='Baseline']
    labels = driver_df['Driver'].tolist()
    values = driver_df['Total Contribution'].tolist()

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=.4, pull=[0.02] * len(labels),
        textinfo='percent+label', hoverinfo='label+percent+value',
        marker=dict(line=dict(color='#FFFFFF', width=2)), sort=False
    ))
    fig.update_layout(
        title=dict(text="Pie Chart of Media Contributions", y=0.95),
        showlegend=True, legend=dict(title='<b>Media Drivers</b>', orientation='v', font=dict(size=10)),
        font=dict(family="sans-serif", size=11),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=500, margin=dict(l=20, r=20, t=80, b=20)
    )
    return fig

def budget_bounds_plot(df):
    df_sorted = df.sort_values("optimized_spend_by_period", ascending=False).reset_index(drop=True)
    upper_err = df_sorted["upper_bound"] - df_sorted["optimized_spend_by_period"]
    lower_err = df_sorted["optimized_spend_by_period"] - df_sorted["lower_bound"]
    
    bar_trace = go.Bar(
        x=df_sorted["index"], y=df_sorted["optimized_spend_by_period"], name="Optimized Spend",
        marker_color="cornflowerblue",
        error_y=dict(type="data", symmetric=False, array=upper_err, arrayminus=lower_err, color="red", thickness=2, width=8)
    )
    fig = go.Figure(data=[bar_trace])
    fig.update_layout(title="Average Optimized Spend & Budget Bounds", xaxis_title="Media", yaxis_title="Spend", bargap=0.3, template="simple_white", height=600)
    return fig

def bubble_plot(df):
    df_cleaned = df.rename(columns=lambda c: c.strip())
    fig = px.scatter(
        df_cleaned, x='media_spend', y='contribution', size='ROI', color='channel', hover_name='channel', size_max=60,
        labels={'media_spend': 'Media Spend', 'contribution': 'Contribution', 'ROI': 'ROI'}
    )
    fig.update_traces(marker_symbol='circle')
    fig.update_layout(title='Contribution vs Media Spend (Size = ROAS)', xaxis_title='Media Spend', yaxis_title='Contribution', template='simple_white', height=800)
    return fig
   
def response_curve_plot(df):
    df_geo = df
    agg_data = df_geo.groupby(['channel', 'spend_multiplier']).agg(spend_avg=('media_spend', 'mean'), outcome_avg=('sales', 'mean')).reset_index()
    agg_data = agg_data.sort_values(by=['channel', 'spend_avg'])

    fig = px.line(agg_data, x="spend_avg", y="outcome_avg", color="channel", markers=True, labels={"spend_avg": "Average Media Spend", "outcome_avg": "Average Incremental Outcome"})
    fig.update_layout(title="Aggregated Response Curve", xaxis_tickangle=-45, showlegend=True, template="simple_white", height=800, legend_title="Channel")
    return fig

def combi_plots(df, condi):
    df.columns = df.columns.str.strip()
    if condi == 'Original':
        df['% Spend'] = df['Original Spend'] / df['Original Spend'].sum() * 100
        df['% Revenue'] = df['Original Revenue'] / df['Original Revenue'].sum() * 100
        title_res = 'Spend and Revenue Contribution by Marketing Channel (Original)'
    else:
        df['% Spend'] = df['Optimal Spend'] / df['Optimal Spend'].sum() * 100
        df['% Revenue'] = df['Optimal Revenue'] / df['Optimal Revenue'].sum() * 100
        title_res = 'Spend and Revenue Contribution by Marketing Channel (Optimize)'

    fig = go.Figure()
    if condi == "Original":
        fig.add_trace(go.Bar(x=df['Media Variable'], y=df['% Revenue'], name='% Revenue', marker_color='rgb(31,119,180)', customdata=df['Original Revenue'], hovertemplate="Revenue: $%{customdata:,.0f}<br>Share: %{y:.1f}%<extra></extra>", text=df['% Revenue'].round(1).astype(str), textposition='outside'))
        fig.add_trace(go.Bar(x=df['Media Variable'], y=df['% Spend'], name='% Spend', marker_color='rgb(174,199,232)', customdata=df['Original Spend'], hovertemplate="Spend: $%{customdata:,.0f}<br>Share: %{y:.1f}%<extra></extra>", text=df['% Spend'].round(1).astype(str), textposition='outside'))
    else:
        fig.add_trace(go.Bar(x=df['Media Variable'], y=df['% Revenue'], name='% Revenue', marker_color='rgb(0,100,0)', customdata=df['Optimal Revenue'], hovertemplate="Revenue: $%{customdata:,.0f}<br>Share: %{y:.1f}%<extra></extra>", text=df['% Revenue'].round(1).astype(str), textposition='outside'))
        fig.add_trace(go.Bar(x=df['Media Variable'], y=df['% Spend'], name='% Spend', marker_color='rgb(144,238,144)', customdata=df['Optimal Spend'], hovertemplate="Spend: $%{customdata:,.0f}<br>Share: %{y:.1f}%<extra></extra>", text=df['% Spend'].round(1).astype(str), textposition='outside'))

    roi_y = df['Original ROI'] if condi=='Original' else df['Optimal ROI']
    roi_color = 'red' if condi=='Original' else 'green'
    fig.add_trace(go.Scatter(x=df['Media Variable'], y=roi_y, mode='markers+text', name='ROAS', marker=dict(symbol='line-ew', size=20, color=roi_color, line=dict(width=4)), text=roi_y.round(1).astype(str), textposition='top center', hovertemplate="ROAS: %{y:.1f}<extra></extra>"))
    fig.update_layout(title=title_res, barmode='group', yaxis=dict(title='% Percentage', tickformat='.0f'), template='simple_white', height=600, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig

def inc_rev_plot(df, kpi, pad_pct: float = 0.05):
    if kpi == 'Revenue':
        orig, opt, ytitle, title = df['Original Revenue'], df['Optimal Revenue'], 'Revenue', 'Optimized Incremental Revenue Across All Channels'
    else:
        orig, opt, ytitle, title = df['Original Spend'], df['Optimal Spend'], 'Spend', 'Optimized Incremental Spend Across All Channels'

    baseline = orig.sum()
    deltas = (opt - orig).tolist()
    optimized = baseline + sum(deltas)

    measures = ['absolute'] + ['relative'] * len(deltas) + ['total']
    xs = ['non_optimized'] + df['Media Variable'].tolist() + ['optimized']
    ys = [baseline] + deltas + [optimized]
    texts = [f"{v:,.0f}" for v in ys]

    fig = go.Figure(go.Waterfall(name=kpi, measure=measures, x=xs, y=ys, text=texts, textposition='outside', decreasing={'marker': {'color': 'salmon'}}, increasing={'marker': {'color': 'teal'}}, totals={'marker': {'color': 'royalblue'}}, connector={'line': {'color': 'gray'}}))

    running = list(itertools.accumulate(deltas, initial=baseline))
    ymin, ymax = min(running), max(running)
    span = ymax - ymin
    fig.update_layout(yaxis=dict(range=[ymin - pad_pct*span, ymax + pad_pct*span], tickformat=',', title=ytitle), title=title, waterfallgroupgap=0.3, showlegend=False, template='simple_white', height=600)
    return fig

def popover_filter(channels, key_prefix: str) -> list[str]:
    pop = st.popover("Filter Media Channels")
    if pop.button("Select all", key=f"{key_prefix}_select_all"):
        for ch in channels: st.session_state[f"{key_prefix}_filter_{ch}"] = True
    if pop.button("Unselect all", key=f"{key_prefix}_unselect_all"):
        for ch in channels: st.session_state[f"{key_prefix}_filter_{ch}"] = False

    selected: list[str] = []
    for ch in channels:
        state_key = f"{key_prefix}_filter_{ch}"
        default = st.session_state.get(state_key, True)
        if pop.checkbox(label=ch, value=default, key=state_key):
            selected.append(ch)
    return selected

def popover_filter_geo(geo_data, key_prefix: str) -> list[str]:
    pop = st.popover("Filter Media Geo")
    if pop.button("Select all", key=f"{key_prefix}_select_all"):
        for ch in geo_data: st.session_state[f"{key_prefix}_filter_{ch}"] = True
    if pop.button("Unselect all", key=f"{key_prefix}_unselect_all"):
        for ch in geo_data: st.session_state[f"{key_prefix}_filter_{ch}"] = False

    selected: list[str] = []
    for i, ch in enumerate(geo_data):
        state_key = f"{key_prefix}_filter_{ch}"
        default = st.session_state.get(state_key, (i == 0))
        if pop.checkbox(label=ch, value=default, key=state_key):
            selected.append(ch)
    return selected


# --- Page Configuration ---
st.set_page_config(layout='wide', page_title="MMM Dashboard", page_icon=logo_img, initial_sidebar_state="collapsed")

# --- App Styling ---
if Path(logo_path).is_file():
    logo_base64 = image_to_base64(logo_path)
    st.markdown(f"""
    <style>
    :root {{
        --header-height: 100px; 
        --header-bg: #1e1a34;
        --sidebar-bg: #2A263F;
        --text-color: #FFFFFF;
        --highlight-color: #4B4E6D;
    }}
    .stApp {{ padding-top: 0px !important; }}
    [data-testid="stHeader"] {{
        position: fixed; top: 0; left: 0; width: 100%; height: var(--header-height); background-color: var(--header-bg); z-index: 1000; display: flex; align-items: center; padding: 0 2rem;
    }}
    [data-testid="stSidebar"] {{
        position: fixed; top: var(--header-height); left: 0; height: calc(100% - var(--header-height)); background-color: var(--sidebar-bg);
    }}
    [data-testid="stSidebar"] [data-testid="stBlock"]:first-child {{ margin-top: 0.5rem !important; }}
    [data-testid="stFileUploader"] section {{ background-color: rgba(255, 255, 255, 0.05); border-color: rgba(255, 255, 255, 0.2); }}
    [data-testid="stFileUploader"] section:hover {{ border-color: var(--highlight-color); }}
    [data-testid="stFileUploader"] section small {{ color: rgba(255, 255, 255, 0.5) !important; }}
    [data-testid="stFileUploader"] button {{ background-color: var(--highlight-color); color: white; border: none; }}
    [data-testid="stFileUploader"] button:hover {{ background-color: #5a5d80; color: white; border: none; }}
    [data-testid="stAppViewContainer"] > section {{ padding-top: 10px; }}
    .logo-container {{ position: fixed; top: 20px; left: 2rem; z-index: 1001; }}
    .logo-container img {{ height: 60px; }}
    [data-testid="stHeader"] *, [data-testid="stSidebar"] * {{ color: var(--text-color); }}
    [data-testid="stHeader"] [data-testid="baseLinkButton-header"]:hover {{ color: #E0E0E0; }}
    div[data-baseweb="select"] {{ max-height: 40px; overflow: auto; }}
    
    /* Hides the '<<' close button when the sidebar is OPEN */
    [data-testid="stSidebarCollapsedControl"] {{ display: none !important; visibility: hidden !important; opacity: 0 !important; }}
    
    /* Hides the '>' expand button when the sidebar is CLOSED (This fixes the issue) */
    [data-testid="stExpandSidebarButton"] {{display: none !important; visibility: hidden !important; opacity: 0 !important; }}
    
    /* Hides mobile 'X' */
    section[data-testid="stSidebar"] > div > div:first-child button {{ display: none !important; }}
    </style>

    <div class="logo-container">
        <img src="data:image/svg+xml;base64,{logo_base64}" alt="Logo">
    </div>
    """, unsafe_allow_html=True)
else:
    st.error("Logo file not found. Please make sure 'logo.svg' is in the same directory as your script.")


# --- App Logic & Routing ---
init_session()
org = st.session_state.identity_org
brand = st.session_state.identity_brand

meridian = get_meridian_resource()

st.sidebar.title("🔐 Proof of Concept")
st.sidebar.title("🚀 Get Started: Load Dataset & Model")

with st.sidebar:
    st.header("File Upload")
    with st.expander("Upload a New Dataset", expanded=False):
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")
    
    st.header("Model Upload")
    with st.expander("Upload a New Model", expanded=False):
        uploaded_model = st.file_uploader("Upload Model", type=["pkl"], label_visibility="collapsed")


# Gatekeeper: Halt execution if no valid organization/brand is supplied in URL
if org == "unknown" or brand == "unknown":
    st.title("Welcome to the Scene MX Dashboard")
    st.warning("⚠️ No valid client configuration detected.")
    st.info("Please provide a valid organization and brand in the URL. Example: `/?org=assembly&brand=test_brand`")
    st.stop()

st.title(f"Scene MX Dashboard - {brand.replace('_', ' ').title()}")
st.subheader(f'Welcome, viewing data for {org.upper()}')


# --- Load Dataset Logic ---
if 'dataset' not in st.session_state:
    st.session_state.dataset = None

if uploaded_file is not None:
    with st.spinner("Loading uploaded dataset..."):
        try:
            st.session_state.dataset = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            st.success("✅ Successfully loaded uploaded dataset!")
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")
elif st.session_state.dataset is None:
    with st.spinner(f"Loading data for {org}..."):
        try:
            st.session_state.dataset = load_client_dataset(org, brand)
        except Exception as e:
            st.error(f"🚨 Default Dataset Load Failed: {e}")

dataset = st.session_state.dataset

if dataset is not None:
    dt, date_column, geo_option, y_col, paid_media_vars, non_paid_media_vars, control_vars, date_start, date_end, spend_mul_inc, n_reference, inputs, total_budget, maxiter, popsize = selector_filter(uploaded_file, dataset)
else:
    st.warning("Please upload a dataset or ensure the valid client config is set.")
    st.stop() 


# --- Load Model Logic ---
if 'smmm_model' not in st.session_state:
    st.session_state.smmm_model = None

if uploaded_model is not None:
    with st.spinner("Loading uploaded model..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            temp_file.write(uploaded_model.getvalue())
            temp_file_path = temp_file.name 
        try:
            st.session_state.smmm_model = meridian.model.load_mmm(temp_file_path)
            if "historical_full_df" in st.session_state:
                del st.session_state.historical_full_df
            st.success("✅ Successfully loaded uploaded model!")
        finally:
            os.remove(temp_file_path)
elif st.session_state.smmm_model is None:
    with st.spinner(f"Loading model for {org}..."):
        try:
            st.session_state.smmm_model = load_client_model(org, brand)
            if "historical_full_df" in st.session_state:
                del st.session_state.historical_full_df
        except Exception as e:
            st.error(f"🚨 Default Model Load Failed: {e}")

smmm = st.session_state.smmm_model

if smmm is None:
    st.warning("Cannot proceed: A model is required. Please upload one or ensure the valid client config is set.")
    st.stop() 

# --- Session Variables Initialize ---
for key in ["opt_caption", "opt_media_result", "opt_budget_bounds", "opt_media_summary", "opt_paid_media_vars", 
            "opt_geo_option", "opt_driver", "opt_response", "opt_non_paid_media_vars", "opt_control_media_vars", 
            "opt_max_deviation", "forecast_func"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.write()

# --- Execution Core ---
run_clicked = st.button("🏃💨 Run Budget Allocation Optimization", type="primary")

if st.session_state.opt_caption is not None:
    st.caption(st.session_state.opt_caption)
else:
    st.write("")
     
plot_area = st.empty() 

if run_clicked:
    with st.status("Executing the Model - Please Don't Change Anything Yet...") as status:
        plot_area.empty()
        budget_optimizer = meridian.optimizer.BudgetOptimizer(smmm)
        geos_opt = smmm.input_data.geo.values.tolist() 
        selected_geo = geo_option 
        set_geos_opt = set(geos_opt)
        geos_opt = [item for item in selected_geo if item in set_geos_opt]

        main_media_vars = smmm.input_data.media_channel.values.flatten().tolist()  
        main_media_vars = dict.fromkeys(main_media_vars, 0)
        
        paid_media_bounds_dictionary = main_media_vars.copy()
        paid_media_bounds_dictionary.update(inputs) 
        
        optimization_start_date = date_start.strftime("%Y-%m-%d")
        optimization_end_date = date_end.strftime("%Y-%m-%d")
        optimization_date_range = (optimization_start_date, optimization_end_date)
        geo_opt_dict = {}

        mmm_paid_media = list(paid_media_bounds_dictionary.keys())
        mmm_paid_media_values = list(paid_media_bounds_dictionary.values())
        spend_constraint_lower = mmm_paid_media_values         
        spend_constraint_upper = mmm_paid_media_values

        step_fraction = n_reference/100 
        min_step = 50
        max_step = 5000

        for geo in geos_opt:
            spend_data = smmm.input_data.media_spend.sel(geo=geo, time=slice(optimization_date_range[0], optimization_date_range[1]))
            total_spend = spend_data.sum(dim='time').values.sum()
            step_size = max(min_step, min(max_step, total_spend * step_fraction))

            print(f"\n{geo.upper()} geo analysis:")
            print("═" * 50 + "\n")
            print(f"Total historical spend: ${total_spend:,.2f}")
            print(f"Step size: ${step_size:,.2f}")

            channel_spend = spend_data.sum(dim='time')
            zero_indexes = np.where(channel_spend.values == 0)[0]

            if total_spend > 0:
                geo_opt_dict[geo] = {
                    'total_spend': total_spend,
                    'spend_constraint_lower': spend_constraint_lower.copy(),
                    'spend_constraint_upper': spend_constraint_upper.copy(),
                    'step_size': step_size,
                }
                if len(zero_indexes) > 0:
                    print(f"\n⚠️  Found {len(zero_indexes)} channels with zero historical spend:")
                    print("───────────────────────────────────────────────")
                    for idx in zero_indexes:           
                        try:
                            geo_opt_dict[geo]['spend_constraint_lower'][idx] = 0
                            geo_opt_dict[geo]['spend_constraint_upper'][idx] = 0
                            print(f"   • {mmm_paid_media[idx]}")
                        except IndexError:
                            pass
                    print("───────────────────────────────────────────────")
            else:
                print(f"\n⚠️  total_spend is 0 for {geo}. Skipping geo optimization.")
        
        t1 = time.time()
        model_analysis = meridian.analyzer.Analyzer(smmm)
        
        start_time = time.perf_counter()
        optimization_date_range = (optimization_start_date, optimization_end_date)
        optimization_results_geo_meridian = {}
        
        for geo_example in geos_opt:
            if geo_example in geo_opt_dict:
                print(f"Optimizing geo: {geo_example}...")
                optimization_results_geo_meridian[geo_example] = budget_optimizer.optimize(
                    start_date=optimization_start_date, end_date=optimization_end_date,
                    selected_geos=[geo_example], fixed_budget=True,
                    spend_constraint_lower=geo_opt_dict[geo_example]['spend_constraint_lower'],
                    spend_constraint_upper=geo_opt_dict[geo_example]['spend_constraint_upper']
                )

        optimized_df_geo = meridian.get_optimized_df(optimization_results_geo_meridian)
        end_time = time.perf_counter()
        print(f"(Sequential) Processing time estimated: {end_time - start_time:.4f} seconds")
        
        if "historical_full_df" not in st.session_state:
            st.session_state.historical_full_df = meridian.get_summary_df(model_analysis, meridian.constants, include_baseline=True)
            
        full_df = st.session_state.historical_full_df
        
        final_year = str(date_end.year) +'-01-01'
        selected_time_periods_repsonse_curve = smmm.input_data.time.values
        selected_time_periods_repsonse_curve = selected_time_periods_repsonse_curve[selected_time_periods_repsonse_curve > final_year]
        
        steps_list = list(np.round(np.arange(0, spend_mul_inc, 0.1), 1))
        
        response_curves_by_geo = meridian.get_response_curves_df(
            model_analysis, geos_selected=geos_opt, spend_multipliers=steps_list, time_slices_dict=None, multiprocess=False
        )
        response_curves_by_geo.rename(columns={'incremental_outcome_mean':'sales', 'spend':'media_spend'}, inplace=True)

        driver_df = full_df.groupby([geo_column, 'channel']).agg({'incremental_outcome':'sum'}).reset_index().sort_values(by='incremental_outcome', ascending=False).reset_index(drop=True)
        driver_df.rename(columns={'channel': 'Driver', 'incremental_outcome': 'Total Contribution'}, inplace=True)

        decomp_df_unscaled = optimized_df_geo.copy()
        decomp_df_unscaled = decomp_df_unscaled[[geo_column, 'channel', 'incremental_outcome', 'spend', 'roi_calculated', 'incremental_outcome_nonoptimized', 'spend_nonoptimized', 'roi_calculated_nonoptimized']]
        decomp_df_unscaled.fillna(0, inplace=True)
        
        media_summary = decomp_df_unscaled[[geo_column, 'channel', 'incremental_outcome', 'spend', 'roi_calculated', 'incremental_outcome_nonoptimized']].copy()
        media_summary['ROI'] = media_summary['incremental_outcome'] / media_summary['spend'] 
        media_summary['ROI'] = np.where(media_summary['spend'] > 0, media_summary['ROI'], 0) 
        media_summary.rename(columns={'incremental_outcome': 'contribution', 'spend': 'media_spend', 'ROI': 'ROI'}, inplace=True)
        media_summary = media_summary[[geo_column, 'channel', 'contribution', 'media_spend', 'ROI']]
        
        media_result = decomp_df_unscaled.copy()
        media_result.rename(columns={
            'channel': 'Media Variable', 'incremental_outcome': 'Optimal Revenue', 'spend': 'Optimal Spend',
            'roi_calculated': 'Optimal ROI', 'incremental_outcome_nonoptimized': 'Original Revenue',
            'spend_nonoptimized': 'Original Spend', 'roi_calculated_nonoptimized': 'Original ROI'}, inplace=True)
        
        media_result['Reference/Forecast Perod'] = f"[{optimization_start_date}, {optimization_end_date}]"
        media_result = media_result[[geo_column,'Media Variable', 'Original Spend',  'Original Revenue', 'Original ROI', 'Optimal Spend', 'Optimal Revenue', 'Optimal ROI', 'Reference/Forecast Perod']]
          
        try:
            st.session_state.opt_media_result = media_result
            st.session_state.opt_media_summary = media_summary
            st.session_state.opt_paid_media_vars = paid_media_vars
            st.session_state.opt_non_paid_media_vars = non_paid_media_vars
            st.session_state.opt_control_media_vars = control_vars
            st.session_state.opt_geo_option = geos_opt
            st.session_state.opt_caption = "⏰ Kindly re-run to apply the changes."
            st.session_state.opt_response = response_curves_by_geo
            st.session_state.opt_driver = driver_df
            st.session_state.forecast_func = "Enable"
        except Exception as e:
            print(f"An error occurred during Optimize Allocation: {e}")
            status.update(label=f"⚠️ An unexpected error occurred in Optimize Allocation: {e}", state="error", expanded=True)
            st.stop()
        
        t2 = time.time()
        status.update(label=f"✅ Model Implementation Complete! Time Execution: {round(t2-t1, 2)} seconds", state="complete", expanded=False)

# --- Display Logic ---
with plot_area.container():
    if st.session_state.opt_media_result is not None:
        
        st.subheader("Media Summary")
        with st.expander("🔼🔽 Media Summary", expanded=True):
            geo_filter = popover_filter_geo(st.session_state.opt_geo_option, key_prefix="Media_Summary")
            st.write("Selected GEO: ", ", ".join(geo_filter))
            media_summ = st.session_state.opt_media_summary
            media_summ = media_summ.loc[media_summ[geo_column].isin(geo_filter)]
            st.dataframe(media_summ, use_container_width=True, hide_index=True)
        
        st.subheader("Optimal Budget Summary")
        with st.expander("🔼🔽 Optimal Budget Table", expanded=True):
            geo_filter = popover_filter_geo(st.session_state.opt_geo_option, key_prefix="Media_Result_Summary")
            st.write("Selected GEO: ", ", ".join(geo_filter))
            media_result_summ = st.session_state.opt_media_result.loc[st.session_state.opt_media_result['Media Variable'].isin(st.session_state.opt_paid_media_vars)]
            media_result_summ = media_result_summ.loc[media_result_summ[geo_column].isin(geo_filter)]
            st.dataframe(media_result_summ, use_container_width=True, hide_index=True)
            
            a, b, c, d = st.columns(4)
            a.metric("💰 Total Original Spend", f"{media_result_summ['Original Spend'].sum():,.2f}", border=True)
            b.metric("💸 Total Optimize Spend", f"{media_result_summ['Optimal Spend'].sum():,.2f}", border=True)
            c.metric("📉 Total Original Revenue", f"{media_result_summ['Original Revenue'].sum():,.2f}", border=True)
            d.metric("📈 Total Optimize Revenue", f"{media_result_summ['Optimal Revenue'].sum():,.2f}", border=True)
        
        st.subheader("Optimized Incremental Revenue Across All Channels")
        with st.expander("🔼🔽 Incremental Revenue", expanded=True):
            geo_filter = popover_filter_geo(st.session_state.opt_geo_option, key_prefix="Incremental_Revenue_Summary")
            st.write("Selected GEO: ", ", ".join(geo_filter))
            inc_rev_summ = st.session_state.opt_media_result.loc[st.session_state.opt_media_result['Media Variable'].isin(st.session_state.opt_paid_media_vars)]
            inc_rev_summ = inc_rev_summ.loc[inc_rev_summ[geo_column].isin(geo_filter)]
            inc_rev_summ = inc_rev_summ.groupby('Media Variable').agg({'Original Spend':'sum', 'Original Revenue':'sum', 'Optimal Spend':'sum', 'Optimal Revenue':'sum', 'Reference/Forecast Perod':'max'}).reset_index()
            inc_rev_summ['Original ROI'] = inc_rev_summ['Original Revenue'] / inc_rev_summ['Original Spend']
            inc_rev_summ['Optimal ROI'] = inc_rev_summ['Optimal Revenue'] / inc_rev_summ['Optimal Spend']
            
            col1, col2 = st.columns((1, 7))
            with col1: kpi = st.radio("Choose Metric to Display", ["Revenue", "Spend"], key='kpi_1')
            with col2: st.plotly_chart(inc_rev_plot(inc_rev_summ,kpi))
        
        st.subheader("Spend and Revenue Contribution by Marketing Channel")
        with st.expander("🔼🔽 Spend vs Revenue", expanded=True):
            geo_filter = popover_filter_geo(st.session_state.opt_geo_option, key_prefix="Spend_Revenue_Summary")
            st.write("Selected GEO: ", ", ".join(geo_filter))
            spend_rev_summ = st.session_state.opt_media_result.loc[st.session_state.opt_media_result['Media Variable'].isin( st.session_state.opt_paid_media_vars)]
            spend_rev_summ = spend_rev_summ.loc[spend_rev_summ[geo_column].isin(geo_filter)]
            spend_rev_summ = spend_rev_summ.groupby('Media Variable').agg({'Original Spend':'sum', 'Original Revenue':'sum', 'Optimal Spend':'sum', 'Optimal Revenue':'sum', 'Reference/Forecast Perod':'max'}).reset_index()
            spend_rev_summ['Original ROI'] = spend_rev_summ['Original Revenue'] / spend_rev_summ['Original Spend']
            spend_rev_summ['Optimal ROI'] = spend_rev_summ['Optimal Revenue'] / spend_rev_summ['Optimal Spend']
            
            col3, col4 = st.columns((1, 7))
            with col3: kpi_2 = st.radio("Choose Metric to Display", ["Optimize", "Original"], key="kpi_2")
            with col4: st.plotly_chart(combi_plots(spend_rev_summ,kpi_2))
        
        st.subheader("Visualizing Media ROI: Contribution Build-up & Flexible Budget Limits")
        with st.expander("🔼🔽 Waterfall Total Spend Contribution and Budget Bound Analysis", expanded=True):
            col1, col2, col3 = st.columns((1, 1, 4))
            with col1: geo_filter_waterfall = popover_filter_geo(st.session_state.opt_geo_option, key_prefix="Spend_Contribution_Summary")
            with col2: channel_filter = popover_filter(['Baseline'] + st.session_state.opt_paid_media_vars + st.session_state.opt_non_paid_media_vars + st.session_state.opt_control_media_vars + ["Total"], key_prefix="waterfall_bounds")
            st.write("Selected GEO: ", ", ".join(geo_filter_waterfall))
            
            spend_contri_summary = st.session_state.opt_driver.loc[st.session_state.opt_driver['Driver'].isin(channel_filter)]
            spend_contri_summary = spend_contri_summary.loc[spend_contri_summary[geo_column].isin(geo_filter_waterfall)]
            spend_contri_summary = spend_contri_summary.groupby('Driver').agg({'Total Contribution':'sum'}).reset_index()
            
            col1, col2 = st.columns((1, 1))
            with col1:
                try: st.plotly_chart(waterfall_plot(spend_contri_summary))
                except Exception as e: print(f"Waterfall error: {e}"); st.write(" ")
            with col2:
                try: st.plotly_chart(pie_plot(spend_contri_summary))
                except Exception as e: print(f"Pie error: {e}"); st.write(" ")
        
        st.subheader("Contribution vs Media Spend : Bubble Size Indicator For Channel’s ROI")
        with st.expander("🔼🔽 Bubble Chart of Contribution & Media Spend", expanded=True): 
            col1, col2, col3 = st.columns((1, 1, 4))
            with col1: geo_filter_cc = popover_filter_geo(st.session_state.opt_geo_option, key_prefix="Chart_Contribution_Channels")
            with col2: channel_filter_bubble = popover_filter(st.session_state.opt_paid_media_vars, key_prefix="bubble_bounds")
            st.write("Selected GEO: ", ", ".join(geo_filter_cc))
            
            chart_contrib_summ = st.session_state.opt_media_summary.loc[(st.session_state.opt_media_summary[geo_column].isin(geo_filter_cc)) & (st.session_state.opt_media_summary['channel'].isin(channel_filter_bubble))]
            chart_contrib_summ = chart_contrib_summ.groupby('channel').agg({'contribution':'sum', 'media_spend':'sum'}).reset_index()
            chart_contrib_summ['ROI'] = chart_contrib_summ['contribution'] / chart_contrib_summ['media_spend']
            st.plotly_chart(bubble_plot(chart_contrib_summ))

        st.subheader("Sales Response to Media Spend")
        with st.expander("🔼🔽 Response Curve Analysis", expanded=True):
            col1, col2, col3 = st.columns((1, 1, 4))
            with col1: geo_filter_rc = popover_filter_geo(st.session_state.opt_geo_option, key_prefix="Sales_Response_Curves")
            with col2: channel_filter_rc = popover_filter(st.session_state.opt_paid_media_vars, key_prefix="response_curve")
            st.write("Selected GEO: ", ", ".join(geo_filter_rc))

            sales_resp_curv_summ = st.session_state.opt_response.loc[st.session_state.opt_response['channel'].isin(channel_filter_rc)]
            sales_resp_curv_summ = sales_resp_curv_summ.loc[sales_resp_curv_summ[geo_column].isin(geo_filter_rc)]
            
            if len(sales_resp_curv_summ[geo_column].unique()) > 1:
                sales_resp_curv_summ = sales_resp_curv_summ.groupby(['channel', 'spend_multiplier']).agg({'media_spend': 'sum', 'incremental_outcome_ci_hi': 'sum', 'incremental_outcome_ci_lo': 'sum', 'sales': 'sum'}).reset_index()
                sales_resp_curv_summ['geo'] = 'Combined_Geo'
            elif len(sales_resp_curv_summ[geo_column].unique()) > 0 & len(sales_resp_curv_summ[geo_column].unique()) < 1:
                sales_resp_curv_summ = sales_resp_curv_summ.groupby(['channel', 'spend_multiplier']).agg({'media_spend': 'sum', 'incremental_outcome_ci_hi': 'sum', 'incremental_outcome_ci_lo': 'sum', 'sales': 'sum'}).reset_index()
                sales_resp_curv_summ['geo'] = geo_filter_rc[0]
            
            st.plotly_chart(response_curve_plot(sales_resp_curv_summ))
            
    else:
        st.write("Click ▶️ to run budget allocation optimization.")

    st.divider()
    
    # --- Forecasting Engine ---
    if st.session_state.forecast_func is not None:
        st.subheader("Budget Optimization Simulation")
        
        time_coords = pd.to_datetime(smmm.input_data.time.values)
        min_model_date = time_coords.min().date()
        max_model_date = time_coords.max().date()

        safe_default_start = max(min_model_date, date_start)
        safe_default_end = min(max_model_date, date_end)

        if safe_default_start > safe_default_end:
            safe_default_start = min_model_date
            safe_default_end = max_model_date

        forecast_date_range = st.slider(
            label="Simulation Date Range", min_value=min_model_date, max_value=max_model_date,
            value=(safe_default_start, safe_default_end), format="YYYY-MM-DD", on_change=collapse_forecast_expander
        )
        
        run_forecast = st.button("🏃‍📈 Run Budget Simulation Optimization", type="primary")
        
        if "total_result_future" not in st.session_state:
            st.session_state.total_result_future = None
        
        if run_forecast:
            with st.status("Executing the Model - Please Don't Change Anything Yet...") as status_forecast:
                budget_optimizer = meridian.optimizer.BudgetOptimizer(smmm)
                geos_opt = smmm.input_data.geo.values.tolist() 
                selected_geo = geo_option 
                set_geos_opt = set(geos_opt)
                geos_opt = [item for item in selected_geo if item in set_geos_opt]
        
                main_media_vars = smmm.input_data.media_channel.values.flatten().tolist()  
                main_media_vars = dict.fromkeys(main_media_vars, 0) 
                
                paid_media_bounds_dictionary = main_media_vars.copy() 
                paid_media_bounds_dictionary.update(inputs) 
                
                optimization_start_date = forecast_date_range[0].strftime("%Y-%m-%d")
                optimization_end_date = forecast_date_range[1].strftime("%Y-%m-%d")
                optimization_date_range = (optimization_start_date, optimization_end_date)
        
                geo_opt_dict = {}
                mmm_paid_media = list(paid_media_bounds_dictionary.keys())
                mmm_paid_media_values = list(paid_media_bounds_dictionary.values())
        
                spend_constraint_lower = mmm_paid_media_values         
                spend_constraint_upper = mmm_paid_media_values
                step_fraction = n_reference/100 
                min_step, max_step = 50, 5000
        
                for geo in geos_opt:                                                        
                    spend_data = smmm.input_data.media_spend.sel(geo=geo, time=slice(optimization_date_range[0], optimization_date_range[1]))
                    total_spend = spend_data.sum(dim='time').values.sum()
                    step_size = max(min_step, min(max_step, total_spend * step_fraction))
        
                    channel_spend = spend_data.sum(dim='time')
                    zero_indexes = np.where(channel_spend.values == 0)[0]                   
        
                    if total_spend > 0:
                        geo_opt_dict[geo] = {
                            'total_spend': total_spend, 'spend_constraint_lower': spend_constraint_lower.copy(),
                            'spend_constraint_upper': spend_constraint_upper.copy(), 'step_size': step_size,
                        }
                        if len(zero_indexes) > 0:
                            for idx in zero_indexes:           
                                try:                                  
                                    geo_opt_dict[geo]['spend_constraint_lower'][idx] = 0
                                    geo_opt_dict[geo]['spend_constraint_upper'][idx] = 0
                                except IndexError: pass
        
                t1 = time.time()
                model_analysis = meridian.analyzer.Analyzer(smmm)
                
                start_time = time.perf_counter()
                optimization_results_geo_meridian = {}
                
                for geo_example in geos_opt:
                    if geo_example in geo_opt_dict:
                        optimization_results_geo_meridian[geo_example] = budget_optimizer.optimize(
                            start_date=optimization_start_date, end_date=optimization_end_date, selected_geos=[geo_example], fixed_budget=True,
                            spend_constraint_lower=geo_opt_dict[geo_example]['spend_constraint_lower'],
                            spend_constraint_upper=geo_opt_dict[geo_example]['spend_constraint_upper']
                        )

                optimized_df_geo = meridian.get_optimized_df(optimization_results_geo_meridian)
                end_time = time.perf_counter()
                
                if "historical_full_df" not in st.session_state:
                    st.session_state.historical_full_df = meridian.get_summary_df(model_analysis, meridian.constants, include_baseline=True)
                full_df = st.session_state.historical_full_df
                
                driver_df = full_df.groupby([geo_column, 'channel']).agg({'incremental_outcome':'sum'}).reset_index().sort_values(by='incremental_outcome', ascending=False).reset_index(drop=True)
                driver_df.rename(columns={'channel': 'Driver', 'incremental_outcome': 'Total Contribution'}, inplace=True)
        
                decomp_df_unscaled = optimized_df_geo.copy()
                decomp_df_unscaled = decomp_df_unscaled[[geo_column, 'channel', 'incremental_outcome', 'spend', 'roi_calculated', 'incremental_outcome_nonoptimized', 'spend_nonoptimized', 'roi_calculated_nonoptimized']]
                decomp_df_unscaled.fillna(0, inplace=True)  
                
                media_summary = decomp_df_unscaled[[geo_column, 'channel', 'incremental_outcome', 'spend', 'roi_calculated', 'incremental_outcome_nonoptimized']].copy()
                media_summary['ROI'] = media_summary['incremental_outcome'] / media_summary['spend'] 
                media_summary['ROI'] = np.where(media_summary['spend'] > 0, media_summary['ROI'], 0) 
                media_summary.rename(columns={'incremental_outcome': 'contribution', 'spend': 'media_spend', 'ROI': 'ROI'}, inplace=True)
                media_summary = media_summary[[geo_column, 'channel', 'contribution', 'media_spend', 'ROI']]
                
                media_result = decomp_df_unscaled.copy()
                media_result.rename(columns={
                    'channel': 'Media Variable', 'incremental_outcome': 'Optimal Revenue', 'spend': 'Optimal Spend',
                    'roi_calculated': 'Optimal ROI', 'incremental_outcome_nonoptimized': 'Original Revenue', 'spend_nonoptimized': 'Original Spend',
                    'roi_calculated_nonoptimized': 'Original ROI'}, inplace=True)
                media_result['Reference/Forecast Perod'] = f"[{optimization_start_date}, {optimization_end_date}]"
                media_result = media_result[[geo_column,'Media Variable', 'Original Spend',  'Original Revenue', 'Original ROI', 'Optimal Spend', 'Optimal Revenue', 'Optimal ROI', 'Reference/Forecast Perod']]
                  
                try:
                    st.session_state.forecast_media_result = media_result
                    st.session_state.forecast_media_summary = media_summary
                    st.session_state.forecast_paid_media_vars = paid_media_vars
                    st.session_state.forecast_non_paid_media_vars = non_paid_media_vars
                    st.session_state.forecast_control_media_vars = control_vars
                    st.session_state.forecast_geo_option = geos_opt
                    st.session_state.forecast_driver = driver_df
                    st.session_state.total_result_future = "Enable"
                    st.session_state.forecast_expander_state = True
                except Exception as e:
                    status_forecast.update(label=f"⚠️ An unexpected error occurred: {e}", state="error", expanded=True)
                    st.stop()
                
                t2 = time.time()
                status_forecast.update(label=f"✅ Model Implementation Complete! Time Execution: {round(t2-t1, 2)} seconds", state="complete", expanded=False)
        
        if 'forecast_expander_state' not in st.session_state:
            st.session_state.forecast_expander_state = True 
        
        if st.session_state.total_result_future is not None:    
            with st.expander("Forecast Result: ", expanded=st.session_state.forecast_expander_state):
                st.html("<h3> Predicted Total Result </h3>")
                geo_filter_forecast = popover_filter_geo(st.session_state.forecast_geo_option, key_prefix="Media_Result_Forecaset_Summary")
                st.write("Selected GEO: ", ", ".join(geo_filter_forecast))
                
                media_result_summ_forecast = st.session_state.forecast_media_result.loc[st.session_state.forecast_media_result['Media Variable'].isin(st.session_state.forecast_paid_media_vars)]
                media_result_summ_forecast = media_result_summ_forecast.loc[media_result_summ_forecast[geo_column].isin(geo_filter_forecast)]
                
                a, b, c, d, e, f = st.columns([1.25, 1.25, 1.25, 1.25, 1, 1])
                
                total_ori_spend = media_result_summ_forecast['Original Spend'].sum()
                total_opt_spend = media_result_summ_forecast['Optimal Spend'].sum()
                total_ori_revenue = media_result_summ_forecast['Original Revenue'].sum()
                total_opt_revenue = media_result_summ_forecast['Optimal Revenue'].sum()
                
                total_ori_roi_forecast = total_ori_revenue / total_ori_spend if total_ori_spend > 0 else 0
                total_opt_roi_forecast = total_opt_revenue / total_opt_spend if total_opt_spend > 0 else 0
                
                a.metric("🌱 Total Original Spend", f"{total_ori_spend:,.2f}", border=True)
                b.metric("🎯 Total Optimize Spend", f"{total_opt_spend:,.2f}", border=True)
                c.metric("📈 Total Original Revenue", f"{total_ori_revenue:,.2f}", border=True)
                d.metric("🚀 Total Optimize Revenue", f"{total_opt_revenue:,.2f}", border=True)
                e.metric("💹 Total Original ROAS", f"{total_ori_roi_forecast:,.2f}", border=True)
                f.metric("💎 Total Optimize ROAS", f"{total_opt_roi_forecast:,.2f}", border=True)
                
                st.markdown('<hr style="height:1px;border:none;background-color:green;" />', unsafe_allow_html=True)
                st.html("<h3> Predicted Media Result </h3>")
                
                spend_rev_forecast_summ = st.session_state.forecast_media_result.loc[st.session_state.forecast_media_result['Media Variable'].isin( st.session_state.forecast_paid_media_vars)]
                spend_rev_forecast_summ = spend_rev_forecast_summ.loc[spend_rev_forecast_summ[geo_column].isin(geo_filter_forecast)]
                spend_rev_forecast_summ = spend_rev_forecast_summ.groupby('Media Variable').agg({'Original Spend':'sum', 'Original Revenue':'sum', 'Optimal Spend':'sum', 'Optimal Revenue':'sum', 'Reference/Forecast Perod':'max'}).reset_index()
                spend_rev_forecast_summ['Original ROI'] = spend_rev_forecast_summ['Original Revenue'] / spend_rev_forecast_summ['Original Spend']
                spend_rev_forecast_summ['Optimal ROI'] = spend_rev_forecast_summ['Optimal Revenue'] / spend_rev_forecast_summ['Optimal Spend']
            
                col3, col4 = st.columns((1, 7))
                with col3: kpi_2_forecast = st.radio("Choose Metric to Display", ["Optimize", "Original"], key="kpi_2_forecast")
                with col4: st.plotly_chart(combi_plots(spend_rev_forecast_summ, kpi_2_forecast), key="forecast_combi_plot_unique_id")
                
                df_media_result_future = st.checkbox("🔼🔽 Show Media Result Forecast Table", key="df_media_result_future")
                if df_media_result_future:
                    st.dataframe(st.session_state.forecast_media_result, hide_index=True, use_container_width=True, column_config={"Reference/Forecast Period": None})