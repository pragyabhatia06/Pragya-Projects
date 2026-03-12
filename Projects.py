import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Pragya Bhatia | Data Engineering Portfolio",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Pragya Bhatia | Data Engineering Portfolio")
    st.caption(
        "Interactive portfolio of data engineering, cloud optimization, SQL performance, GIS analytics, NLP pipelines, and monitoring dashboards."
    )

    st.sidebar.success("Select a project from the sidebar.")

    st.markdown(
        """
Welcome to my portfolio of **data engineering, analytics, automation, and AI-driven applications**.

This workspace showcases projects built around:

- **ETL / ELT pipeline design**
- **Data Quality and Validation**
- **SQL Optimization**
- **Cloud Cost Monitoring**
- **Web scraping and API ingestion**
- **GIS and geospatial analytics**
- **NLP and ML-powered data applications**

---
"""
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Experience", "10+ Years")
    k2.metric("Core Stack", "Python • SQL • PostgreSQL")
    k3.metric("Cloud", "AWS • Azure • BigQuery")
    k4.metric("Focus", "Data Platforms & Optimization")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚀 Featured Data Engineering Projects")
        st.markdown(
            """
1. **Web Scraping Data Pipeline**  
   End-to-end ingestion pipeline with scraping, transformation, loading, and monitoring.

2. **Data Pipeline Monitoring Dashboard**  
   Production-style pipeline health tracking with SLA breaches, failures, and run analytics.

3. **Cloud Cost Optimization Dashboard**  
   Analyze cloud spend, detect cost-saving opportunities, and estimate projected savings.

4. **Data Quality & Validation Dashboard**  
   Profile datasets, detect nulls, duplicates, invalid values, and export cleaned data.

5. **SQL Query Performance Analyzer**  
   Detect anti-patterns, review joins/filters, and suggest optimization improvements.

6. **GIS Data Engineering Dashboard**  
   Geospatial analytics for ports, route calculations, coordinate validation, and GIS reporting.
"""
        )

    with col2:
        st.subheader("🤖 AI / Analytics / ML Projects")
        st.markdown(
            """
7. **Translation Pipeline Dashboard**  
   Translation workflow with audit logging, monitoring, and pipeline-style observability.

8. **Sentiment Analysis Pipeline**  
   Text analytics dashboard with single/batch sentiment processing and monitoring.

9. **Llama2 QnA Chatbot**  
   LLM-based QnA assistant demo using Hugging Face / Meta-based workflow.

10. **Monkey Pox Prediction Pipeline**  
    ML prediction demo with structured input flow and model inference.

11. **Interactive Visualization & Analytics Demos**  
    Supporting demos for charts, plotting, and analytical presentation.
"""
        )

    st.markdown("---")

    st.subheader("🧩 Current Portfolio Navigation")
    st.markdown(
        """
Use the **left sidebar** to open each live project page.

### Suggested viewing order
- Web Scraping Data Pipeline
- Data Pipeline Monitoring Dashboard
- Cloud Cost Optimization Dashboard
- Data Quality & Validation Dashboard
- SQL Query Performance Analyzer
- GIS Data Engineering Dashboard
"""
    )

    st.markdown("---")

    st.subheader("💼 Professional Profiles")
    st.markdown(
        """
- [LinkedIn](https://www.linkedin.com/in/pragya-bhatia/)
- [GitHub](https://github.com/pragyabhatia06)
- [HackerRank](https://www.hackerrank.com/profile/pragya_bhatia_06)
"""
    )

    st.subheader("📜 Certifications")
    st.markdown(
        """
- [Microsoft Azure Data Fundamentals](https://www.credly.com/badges/aab2fdf4-c454-49fe-9f40-7d1793afbed0/public_url)
- [Microsoft Azure Fundamentals](https://www.credly.com/badges/7a66102f-5a98-437c-99d2-31744bef91a7/public_url)
- [Microsoft Azure AI Fundamentals](- [Microsoft Azure AI Fundamentals](https://www.credly.com/badges/7a66102f-5a98-437c-99d2-31744bef91a7/public_url)
)
"""
    )

    st.markdown("---")

    st.subheader("🛠️ About This Portfolio")
    st.markdown(
        """
This portfolio is designed to demonstrate practical, business-oriented engineering work rather than only notebook-style experiments.

The projects focus on:
- Building reliable data systems
- Improving performance and cost efficiency
- Validating and transforming data
- Supporting analytics and decision-making
- Showcasing production-style dashboards and monitoring workflows
"""
    )

    st.info(
        "Tip: Start with the first 6 projects in the sidebar for the strongest Data Engineering portfolio view."
    )


if __name__ == "__main__":
    run()