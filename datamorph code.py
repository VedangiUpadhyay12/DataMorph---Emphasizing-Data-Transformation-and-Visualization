import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
from io import BytesIO

# Global DataFrame and Summary
df = None
summary_log = []
visualization_images = []  # Store visualizations

# App title
st.title("DataMorph - Emphasizing Data Transformation and Visualization")

# Sidebar for file upload
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a file (CSV or Excel)", type=["csv", "xlsx"])

# Add color picker
st.sidebar.title("Customization")
custom_color = st.sidebar.color_picker("Pick a color for the plot", "#1f77b4")

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
        summary_log.append(f"File uploaded: {uploaded_file.name} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    except Exception as e:
        st.sidebar.error(f"An error occurred while uploading: {e}")

if df is not None:
    # Tabs for navigation
    tabs = st.tabs(["Data Preview", "Visualization", "Summary", "Generate Report"])

    with tabs[0]:  # Data Preview
        st.subheader("Data Preview")
        st.dataframe(df)

        # Data cleaning
        if st.button("Clean Data"):
            initial_shape = df.shape
            df.dropna(inplace=True)
            cleaned_shape = df.shape
            st.success("Missing values dropped!")
            summary_log.append(
                f"Cleaned data: Dropped missing values. Rows reduced from {initial_shape[0]} to {cleaned_shape[0]}."
            )
            st.dataframe(df)

        # Data transformation
        if st.button("Transform Data"):
            le = LabelEncoder()
            transformed_columns = []
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = le.fit_transform(df[col])
                transformed_columns.append(col)
            if transformed_columns:
                st.success("Categorical data encoded!")
                summary_log.append(f"Transformed data: Encoded columns - {', '.join(transformed_columns)}.")
            else:
                st.info("No categorical columns found for encoding.")
            st.dataframe(df)

    with tabs[1]:  # Visualization
        st.subheader("Visualization")
        visualization_option = st.selectbox(
            "Choose a visualization",
            [
                "None",
                "Histogram",
                "Correlation Heatmap",
                "Boxplot",
                "Pairplot",
                "Scatter Plot",
                "Bar Chart",
            ],
        )

        if visualization_option != "None":
            try:
                buf = BytesIO()  # Buffer to save the visualization
                if visualization_option == "Histogram":
                    col = st.selectbox("Select a column", df.select_dtypes(include=[np.number]).columns)
                    bins = st.slider("Number of bins", 5, 50, 10)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    df[col].hist(bins=bins, color=custom_color, ax=ax)
                    ax.set_title(f"Histogram of {col}")
                    ax.set_xlabel(col, fontsize=12)
                    ax.set_ylabel("Frequency", fontsize=12)
                    st.pyplot(fig)
                    summary_log.append(f"Generated histogram for column: {col} with custom color.")
                    fig.savefig(buf, format="png")
                    visualization_images.append(("Histogram", buf))

                elif visualization_option == "Correlation Heatmap":
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
                    ax.set_title("Correlation Heatmap")
                    st.pyplot(fig)
                    summary_log.append("Generated correlation heatmap.")
                    fig.savefig(buf, format="png")
                    visualization_images.append(("Correlation Heatmap", buf))

                elif visualization_option == "Boxplot":
                    col = st.selectbox("Select a column", df.columns)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(y=df[col], color=custom_color, ax=ax)
                    ax.set_title(f"Boxplot of {col}")
                    ax.set_ylabel(col, fontsize=12)
                    st.pyplot(fig)
                    summary_log.append(f"Generated boxplot for column: {col} with custom color.")
                    fig.savefig(buf, format="png")
                    visualization_images.append(("Boxplot", buf))

                elif visualization_option == "Pairplot":
                    fig = sns.pairplot(df, plot_kws={"color": custom_color})
                    st.pyplot(fig)
                    summary_log.append("Generated pairplot with custom color.")
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    visualization_images.append(("Pairplot", buf))

                elif visualization_option == "Scatter Plot":
                    cols = df.select_dtypes(include=[np.number]).columns
                    x_col = st.selectbox("X-axis", cols)
                    y_col = st.selectbox("Y-axis", cols)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(df[x_col], df[y_col], color=custom_color)
                    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
                    ax.set_xlabel(x_col, fontsize=12)
                    ax.set_ylabel(y_col, fontsize=12)
                    st.pyplot(fig)
                    summary_log.append(f"Generated scatter plot for {x_col} vs {y_col} with custom color.")
                    fig.savefig(buf, format="png")
                    visualization_images.append(("Scatter Plot", buf))

                elif visualization_option == "Bar Chart":
                    col = st.selectbox("Select a column", df.columns)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    if df[col].dtype == 'object' or df[col].nunique() < 20:
                        df[col].value_counts().plot(kind="bar", color=custom_color, ax=ax)
                        ax.set_title(f"Bar Chart of {col}")
                        ax.set_xlabel(col, fontsize=12)
                        ax.set_ylabel("Frequency", fontsize=12)
                        st.pyplot(fig)
                        summary_log.append(f"Generated bar chart for column: {col} with custom color.")
                        fig.savefig(buf, format="png")
                        visualization_images.append(("Bar Chart", buf))
                    else:
                        st.error("Bar Chart is not suitable for this column.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    with tabs[2]:  # Summary
        st.subheader("Summary of Actions")
        if summary_log:
            for log in summary_log:
                st.write(f"- {log}")
        else:
            st.info("No actions performed yet.")

    with tabs[3]:  # Generate Report
        st.subheader("Generate Report")
        if st.button("Generate Report"):
            try:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                # Title
                pdf.set_font("Arial", style="B", size=16)
                pdf.cell(0, 10, "DataMorph Report", ln=True, align="C")
                pdf.ln(10)

                # Data Information
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(0, 10, "Dataset Information:", ln=True)
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 10, f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
                pdf.multi_cell(0, 10, "Columns and Data Types:")
                for col, dtype in df.dtypes.items():
                    pdf.multi_cell(0, 10, f"- {col}: {dtype}")
                pdf.ln(10)

                # Summary of Actions
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(0, 10, "Summary of Actions:", ln=True)
                pdf.set_font("Arial", size=10)
                for log in summary_log:
                    pdf.multi_cell(0, 10, log)
                pdf.ln(10)

                # Add Visualizations
                pdf.set_font("Arial", style="B", size=12)
                pdf.cell(0, 10, "Visualizations:", ln=True)
                for title, buf in visualization_images:
                    buf.seek(0)  # Reset buffer pointer to the beginning
                    image_path = f"{title}.png"
                    with open(image_path, "wb") as temp_image:
                        temp_image.write(buf.read())
                    pdf.add_page()  # Add a new page for each visualization
                    pdf.set_font("Arial", size=12)
                    pdf.cell(0, 10, title, ln=True)
                    pdf.image(image_path, x=10, y=None, w=190)
                    buf.close()

                # Save the PDF content into a binary stream
                pdf_data = pdf.output(dest="S").encode("latin1")  # Save to binary format

                # Streamlit download button
                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_data,
                    file_name="DataMorph_Report.pdf",
                    mime="application/pdf",
                )
                st.success("Report generated successfully!")
            except Exception as e:
                st.error(f"An error occurred while generating the report: {e}")

else:
    st.sidebar.warning("Please upload a file to proceed.") 