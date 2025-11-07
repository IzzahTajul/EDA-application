import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
import io
import csv

# --- Page Config ---
st.set_page_config(page_title="📊 EDA Dashboard", layout="wide")

# --- CUSTOM STYLES --- 
st.markdown(
    """ 
    <style> body { background-color: #f7f9fc; } .
    main { background-color: #ffffff; padding: 2rem; border-radius: 15px; box-shadow: 0px 0px 15px rgba(0,0,0,0.05); } 
    h1, h2, h3 { color: #2b6cb0; font-family: 'Segoe UI', sans-serif; } .stButton button 
    { background-color: #2b6cb0; color: white; border-radius: 10px; height: 3em; width: 100%; 
    font-size: 16px; font-weight: bold; transition: 0.3s; } .stButton button:hover { background-color: #2c5282; } 
    </style> """, unsafe_allow_html=True
)

# --- Helper function to check numeric columns ---
def is_numeric_column(df, col_name):
    return pd.api.types.is_numeric_dtype(df[col_name])

def convert_strings_to_numbers(df):
    """Convert numeric-looking string columns into numeric dtype safely."""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

# --- Header ---
st.markdown(
    """
    <h1 style='text-align: center; color: #2F4F4F; background-color: #F8F9FA; padding: 15px; border-radius: 10px;'>
        📈 Complete EDA Application
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        background-color: #E6F0FA;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    ">
        <p style='
            color: #2b6cb0;
            font-size: 28px;
            font-weight: 700;
            font-family: "Segoe UI", sans-serif;
            margin: 0;
        '>
            📂 Upload any dataset and visualize it instantly!
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- FILE UPLOAD SECTION ---
uploaded_file = st.file_uploader("📂 Upload Your Data File", type=["csv", "xlsx", "xls", "txt", "json","xml"])

if uploaded_file is not None:
    try:
        file_name = uploaded_file.name.lower()

        # --- Detect and Read File ---
        if file_name.endswith((".csv", ".txt")):
            sample = uploaded_file.read(2048).decode("utf-8", errors="ignore")
            uploaded_file.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
                sep = dialect.delimiter
            except csv.Error:
                sep = ","
            df = pd.read_csv(uploaded_file, sep=sep)
            st.success(f"✅ File uploaded successfully! (Detected separator: '{sep}')")

        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
            st.success("✅ Excel file uploaded successfully!")

        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)
            st.success("✅ JSON file uploaded successfully!")
        
        elif file_name.endswith(".xml"):
            df = pd.read_xml(uploaded_file)
            st.success("✅ XML file uploaded successfully!")

        else:
            st.error("❌ Unsupported file format.")
            st.stop()

        # --- Convert strings to numeric where possible ---
        df = convert_strings_to_numbers(df)

        # --- DATA PREVIEW ---
        st.subheader("🧾 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        # --- ADDITIONAL INFO ---
        st.markdown("---")
        st.subheader("📊 Dataset Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.write("**Total Rows:**", df.shape[0])
        st.write("**Total Columns:**", df.shape[1])
        st.write("**Missing Values:**", df.isnull().sum().sum())
        st.write("**Duplicate Rows:**", df.duplicated().sum())

        st.markdown("---")

        # --- Visualization Section ---
        st.subheader("🎨 Visualization Playground")

         # Identify numeric columns only for Y-axis
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        categorical_columns = df.select_dtypes(exclude='number').columns.tolist()

        if len(numeric_columns) == 0:
            st.warning("⚠️ No numeric columns found — visualizations may not be available.")
            st.stop()

        # Column selectors
        col1, col2 = st.columns(2)
        with col1:
        # ✅ X-axis default: categorical if available, else first numeric
            x_axis = st.selectbox(
                    "Select X-axis Column (Categorical Preferred)",
                    categorical_columns if len(categorical_columns) > 0 else df.columns
                     )
        with col2:
        # ✅ Y-axis restricted to numeric only
             y_axis = st.selectbox("Select Y-axis Column (Numeric Only)", numeric_columns)

        # Color selection
        color = st.color_picker("🎨 Pick a color for the chart(Bar chart,Scatter Plot)", "#1f77b4")
        
        # Create columns for buttons
        col3, col4, col5 = st.columns(3)

        # --- Line Graph ---
        with col3:
            if st.button("📉 Click Here For Line Graph"):
                if not (is_numeric_column(df, x_axis) and is_numeric_column(df,y_axis)):
                    st.error("⚠️ Both X and Y columns must be numeric for a line graph.")
                else:
                    st.write(f"#### Line Graph: {y_axis} vs {x_axis}")

                    fig, ax = plt.subplots()
                
                    # Get data as numpy arrays
                    x = df[x_axis].values
                    y = df[y_axis].values

                    # Create segments between points for gradient coloring
                    points = np.array([x, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    # Normalize color scale by Y values
                    norm = plt.Normalize(y.min(), y.max())
                    cmap = plt.cm.viridis

                    # Create a line collection with color gradient
                    lc = LineCollection(segments, cmap=cmap, norm=norm)
                    lc.set_array(y)
                    lc.set_linewidth(2)

                    # Add collection to axes
                    line = ax.add_collection(lc)

                    # Add markers for each data point
                    ax.scatter(x, y, c=y, cmap=cmap, edgecolors='black', zorder=3)

                    # Add colorbar
                    cbar = plt.colorbar(line, ax=ax)
                    cbar.set_label(y_axis)

                    # Label axis
                    ax.set_xlabel(x_axis, color='green', fontsize=12)
                    ax.set_ylabel(y_axis, color='red', fontsize=12)
                    ax.set_title(f"{y_axis} vs {x_axis} (Color by {y_axis})", color='purple')

                    # Customize axis colors
                    ax.tick_params(axis='x', colors='green')
                    ax.tick_params(axis='y', colors='red')
                    ax.spines['bottom'].set_color('green')
                    ax.spines['left'].set_color('red')

                    # Add grid and background
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.set_facecolor('#f9f9f9')

                    st.pyplot(fig)

        # --- Bar Chart ---
        with col4:
            if st.button("📊 Click Here For Bar Chart"):
                if not is_numeric_column(df, y_axis):
                    st.error("⚠️ Y-axis must be numeric for bar charts.")
                else:
                    st.write(f"#### Bar Chart: {y_axis} vs {x_axis}")
                    fig, ax = plt.subplots()
                    ax.bar(df[x_axis], df[y_axis], color=color)

                    # Label the axis
                    ax.set_xlabel(x_axis, color='green', fontsize=12)
                    ax.set_ylabel(y_axis, color='red', fontsize=12)
                    ax.set_title(f"{y_axis} vs {x_axis} (Color by {y_axis})", color='purple')
 
                    # Customize axis colors
                    ax.tick_params(axis='x', colors='green')
                    ax.tick_params(axis='y', colors='red')
                    ax.spines['bottom'].set_color('green')
                    ax.spines['left'].set_color('red')

                    ax.grid(axis='y', linestyle='--', alpha=0.6)
                    st.pyplot(fig)

        # --- SCATTER PLOT ---
        with col5:
            if st.button("🟣 Scatter Plot"):
                 if not (is_numeric_column(df, x_axis) and is_numeric_column(df, y_axis)):
                    st.error("⚠️ Both X and Y columns must be numeric for a scatter plot.")
                 else:
                    st.markdown(f"#### Scatter Plot — {y_axis} vs {x_axis}")
                    fig, ax = plt.subplots()
                    ax.scatter(df[x_axis], df[y_axis], color=color)
                    ax.set_xlabel(x_axis)
                    ax.set_ylabel(y_axis)
                    ax.set_title(f"{y_axis} vs {x_axis} (Scatter Plot)")
                    ax.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig)

        st.markdown("---")

        # --- HISTOGRAM ---
        st.subheader("📊 Distribution Analysis (Histogram)")

        numeric_columns = df.select_dtypes(include='number').columns

        if len(numeric_columns) > 0:
            selected_col = st.selectbox("Select a numeric column for histogram", numeric_columns)
            bins = st.slider("Number of bins", 5, 50, 20)

            col1,col2,col3 = st.columns([1,2,1])
            
            with col2:
             st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
             fig, ax = plt.subplots(figsize=(2.5, 1.8), dpi=160)  
            
             ax.hist(df[selected_col].dropna(), bins=bins, color='skyblue',
                    edgecolor='black', alpha=0.7)

             # Label and style
             ax.set_xlabel(selected_col, color='green', fontsize=8)
             ax.set_ylabel("Frequency", color='red', fontsize=8)
             ax.set_title(f"Distribution of {selected_col}", color='purple', fontsize=9)
             ax.tick_params(axis='x', colors='green')
             ax.tick_params(axis='y', colors='red')
             ax.spines['bottom'].set_color('green')
             ax.spines['left'].set_color('red')
             ax.grid(True, linestyle='--', alpha=0.5)
             ax.set_facecolor('#f9f9f9')

            st.pyplot(fig, use_container_width=False)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ No numeric columns available for histogram.")

        # --- CORRELATION HEATMAP ---
        st.subheader("🔥 Correlation Heatmap")

        if len(numeric_columns) > 1:
            corr = df[numeric_columns].corr()

            # Center layout
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                 st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)

             # Smaller, clean heatmap
                 fig, ax = plt.subplots(figsize=(3, 2), dpi=160)
                 sns.heatmap(corr, annot=True, cmap="Blues", linewidths=0.4, ax=ax, cbar=True, square=True)

                 ax.set_title("Correlation Heatmap", fontsize=10, color='purple', pad=8)
                 ax.tick_params(axis='x', labelrotation=45, labelsize=7)
                 ax.tick_params(axis='y', labelsize=7)

                 st.pyplot(fig, use_container_width=False)
                 st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Not enough numeric columns to generate a heatmap.")

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")

else:
    st.info("📁 Please upload a data file to start exploring.")

# -------SIGNATURE / FOOTER --------------------
st.markdown(
    '<p class="signature">Made with 🧠 by <b>IZZAHTAJUL😎</b> using Streamlit | 2025 📊 EDA Dashboard</p>',
    unsafe_allow_html=True
)
