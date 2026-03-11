import re
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="SQL Query Performance Analyzer",
    page_icon="🛠️",
    layout="wide",
)


# ---------------------------------------------------
# Safe display helpers
# ---------------------------------------------------
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


def render_html_table(df: pd.DataFrame, height: int = 360):
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
        code {
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(styled_html, unsafe_allow_html=True)


def render_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str):
    if series.empty:
        st.info("No chart data available.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(series.index.astype(str), series.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------
# Sample queries
# ---------------------------------------------------
SAMPLE_QUERIES = {
    "Slow SELECT * with no filter": """
SELECT *
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE YEAR(o.order_date) = 2024
ORDER BY o.order_date DESC;
""",
    "Large aggregation with possible issues": """
SELECT customer_id, COUNT(*) AS total_orders, SUM(order_amount) AS total_value
FROM orders
GROUP BY customer_id
HAVING SUM(order_amount) > 1000
ORDER BY total_value DESC;
""",
    "Nested subquery example": """
SELECT *
FROM products p
WHERE p.product_id IN (
    SELECT product_id
    FROM order_items
    WHERE quantity > 5
)
AND p.category IS NOT NULL;
""",
    "Good filtered query": """
SELECT
    order_id,
    customer_id,
    order_date,
    order_amount
FROM orders
WHERE order_date >= '2024-01-01'
  AND order_date < '2025-01-01'
  AND order_status = 'COMPLETED';
""",
}


# ---------------------------------------------------
# SQL analysis helpers
# ---------------------------------------------------
def clean_sql(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip())


def count_keyword_occurrences(query: str, keyword: str) -> int:
    pattern = rf"\b{re.escape(keyword)}\b"
    return len(re.findall(pattern, query, flags=re.IGNORECASE))


def detect_tables(query: str) -> list[str]:
    tables = set()

    from_matches = re.findall(r"\bFROM\s+([a-zA-Z0-9_.]+)", query, flags=re.IGNORECASE)
    join_matches = re.findall(r"\bJOIN\s+([a-zA-Z0-9_.]+)", query, flags=re.IGNORECASE)

    for tbl in from_matches + join_matches:
        tables.add(tbl)

    return sorted(tables)


def detect_selected_columns(query: str) -> str:
    match = re.search(r"SELECT\s+(.*?)\s+FROM\s", query, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return "Unknown"

    selected = match.group(1).strip()
    return selected


def detect_order_by_columns(query: str) -> list[str]:
    match = re.search(r"\bORDER\s+BY\s+(.*?)(?:\bLIMIT\b|;|$)", query, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []

    raw = match.group(1)
    cols = [c.strip() for c in raw.split(",")]
    return cols


def detect_where_clause(query: str) -> bool:
    return bool(re.search(r"\bWHERE\b", query, flags=re.IGNORECASE))


def detect_limit_clause(query: str) -> bool:
    return bool(re.search(r"\bLIMIT\b", query, flags=re.IGNORECASE))


def detect_group_by(query: str) -> bool:
    return bool(re.search(r"\bGROUP\s+BY\b", query, flags=re.IGNORECASE))


def detect_order_by(query: str) -> bool:
    return bool(re.search(r"\bORDER\s+BY\b", query, flags=re.IGNORECASE))


def detect_subqueries(query: str) -> int:
    return len(re.findall(r"\(\s*SELECT\b", query, flags=re.IGNORECASE))


def detect_functions_in_filters(query: str) -> list[str]:
    issues = []
    patterns = [
        r"\bWHERE\b.*?\bYEAR\s*\(",
        r"\bWHERE\b.*?\bMONTH\s*\(",
        r"\bWHERE\b.*?\bDAY\s*\(",
        r"\bWHERE\b.*?\bUPPER\s*\(",
        r"\bWHERE\b.*?\bLOWER\s*\(",
        r"\bWHERE\b.*?\bCAST\s*\(",
        r"\bWHERE\b.*?\bDATE\s*\(",
    ]

    labels = ["YEAR()", "MONTH()", "DAY()", "UPPER()", "LOWER()", "CAST()", "DATE()"]

    for pattern, label in zip(patterns, labels):
        if re.search(pattern, query, flags=re.IGNORECASE | re.DOTALL):
            issues.append(label)

    return issues


def detect_like_prefix_wildcard(query: str) -> bool:
    return bool(re.search(r"\bLIKE\s+'%[^']*'", query, flags=re.IGNORECASE))


def detect_select_star(query: str) -> bool:
    return bool(re.search(r"SELECT\s+\*", query, flags=re.IGNORECASE))


def detect_distinct(query: str) -> bool:
    return bool(re.search(r"\bSELECT\s+DISTINCT\b", query, flags=re.IGNORECASE))


def detect_or_in_where(query: str) -> bool:
    where_match = re.search(r"\bWHERE\b(.*?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|;|$)", query, flags=re.IGNORECASE | re.DOTALL)
    if not where_match:
        return False
    return " OR " in where_match.group(1).upper()


def detect_join_count(query: str) -> int:
    return count_keyword_occurrences(query, "JOIN")


def detect_union(query: str) -> int:
    return count_keyword_occurrences(query, "UNION")


def estimate_complexity_score(query: str) -> int:
    score = 0
    if detect_select_star(query):
        score += 2
    if detect_join_count(query) >= 2:
        score += 2
    if detect_subqueries(query) >= 1:
        score += 2
    if detect_functions_in_filters(query):
        score += 2
    if not detect_where_clause(query):
        score += 2
    if detect_distinct(query):
        score += 1
    if detect_or_in_where(query):
        score += 1
    if detect_union(query) >= 1:
        score += 2
    if detect_order_by(query):
        score += 1
    if detect_group_by(query):
        score += 1
    return min(score, 10)


def analyze_sql(query: str, db_type: str, table_size: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    cleaned = clean_sql(query)
    query_upper = cleaned.upper()

    issues = []
    recommendations = []

    tables = detect_tables(cleaned)
    selected_cols = detect_selected_columns(cleaned)
    join_count = detect_join_count(cleaned)
    subquery_count = detect_subqueries(cleaned)
    has_where = detect_where_clause(cleaned)
    has_limit = detect_limit_clause(cleaned)
    has_group = detect_group_by(cleaned)
    has_order = detect_order_by(cleaned)
    function_filters = detect_functions_in_filters(cleaned)
    has_select_star = detect_select_star(cleaned)
    has_distinct = detect_distinct(cleaned)
    has_or = detect_or_in_where(cleaned)
    has_prefix_wildcard = detect_like_prefix_wildcard(cleaned)
    unions = detect_union(cleaned)
    complexity_score = estimate_complexity_score(cleaned)

    def add_issue(category: str, severity: str, finding: str, impact: str, recommendation: str):
        issues.append(
            {
                "category": category,
                "severity": severity,
                "finding": finding,
                "impact": impact,
                "recommendation": recommendation,
            }
        )

    def add_recommendation(priority: str, recommendation: str, reason: str):
        recommendations.append(
            {
                "priority": priority,
                "recommendation": recommendation,
                "reason": reason,
            }
        )

    if has_select_star:
        add_issue(
            "Projection",
            "High",
            "Query uses SELECT *",
            "Reads unnecessary columns, increases IO and memory usage.",
            "Select only required columns explicitly.",
        )
        add_recommendation(
            "High",
            "Replace SELECT * with explicit column list",
            "Reduces scanned data and improves maintainability.",
        )

    if not has_where and table_size in ["Medium", "Large", "Very Large"]:
        add_issue(
            "Filtering",
            "High",
            "No WHERE clause detected on a potentially large table.",
            "May trigger full table scan.",
            "Apply partition-friendly or indexed filters where possible.",
        )
        add_recommendation(
            "High",
            "Add selective filters",
            "Reduces rows scanned and improves execution time.",
        )

    if function_filters:
        add_issue(
            "Filtering",
            "High",
            f"Function used in filter condition: {', '.join(function_filters)}",
            "May prevent index or partition pruning.",
            "Rewrite filters to avoid wrapping indexed columns in functions.",
        )
        add_recommendation(
            "High",
            "Rewrite function-based filters into range-based predicates",
            "Allows indexes and partition elimination to work effectively.",
        )

    if has_prefix_wildcard:
        add_issue(
            "Search Predicate",
            "Medium",
            "LIKE with leading wildcard detected.",
            "Can force full scan because normal index lookup is difficult.",
            "Use prefix search, full-text search, or dedicated search structure if applicable.",
        )
        add_recommendation(
            "Medium",
            "Avoid LIKE '%value%' where possible",
            "Improves index usability.",
        )

    if has_distinct:
        add_issue(
            "Deduplication",
            "Medium",
            "DISTINCT detected.",
            "May add unnecessary sort/hash step.",
            "Confirm whether duplicates are caused by joins and remove at source if possible.",
        )
        add_recommendation(
            "Medium",
            "Investigate why DISTINCT is needed",
            "Often hides upstream join or modelling issues.",
        )

    if join_count >= 2:
        add_issue(
            "Join Complexity",
            "Medium",
            f"Multiple joins detected ({join_count}).",
            "Join order, indexing, and data skew may affect performance.",
            "Validate join keys are indexed and join types are appropriate.",
        )
        add_recommendation(
            "High",
            "Index join keys on large tables",
            "Improves merge/hash/nested loop efficiency depending on engine.",
        )

    if subquery_count >= 1:
        add_issue(
            "Subquery",
            "Medium",
            f"Subquery detected ({subquery_count}).",
            "Nested subqueries can be harder for optimizers and may increase cost.",
            "Consider rewriting as JOIN or CTE if it improves readability/performance.",
        )
        add_recommendation(
            "Medium",
            "Review subquery for possible JOIN/CTE rewrite",
            "Can make execution plan more predictable.",
        )

    if has_or:
        add_issue(
            "Predicate Logic",
            "Medium",
            "OR condition detected in WHERE clause.",
            "May reduce index efficiency.",
            "Consider UNION ALL or decomposing logic if performance is poor.",
        )
        add_recommendation(
            "Medium",
            "Review OR predicates",
            "Some engines perform better with separated predicates.",
        )

    if has_group and has_order:
        add_issue(
            "Aggregation/Sorting",
            "Low",
            "GROUP BY and ORDER BY both detected.",
            "Can introduce additional sorting overhead.",
            "Ensure only necessary ordering is applied after aggregation.",
        )
        add_recommendation(
            "Low",
            "Validate ORDER BY necessity after aggregation",
            "Avoid extra sort operations on large result sets.",
        )

    if not has_limit and not has_group and not has_where and table_size in ["Large", "Very Large"]:
        add_issue(
            "Result Control",
            "High",
            "No LIMIT and no selective logic on large dataset.",
            "Could return very large result sets.",
            "Apply LIMIT for exploratory queries or add filter criteria.",
        )
        add_recommendation(
            "High",
            "Add LIMIT for exploration queries",
            "Protects warehouse resources and client rendering.",
        )

    if unions >= 1:
        add_issue(
            "Set Operations",
            "Medium",
            f"UNION detected ({unions}).",
            "UNION removes duplicates and can be more expensive than UNION ALL.",
            "Use UNION ALL if deduplication is not required.",
        )
        add_recommendation(
            "Medium",
            "Prefer UNION ALL where valid",
            "Avoids unnecessary deduplication cost.",
        )

    if not issues:
        add_recommendation(
            "Low",
            "No major anti-patterns detected",
            "Query appears reasonably structured based on heuristic checks.",
        )

    if tables:
        for tbl in tables:
            add_recommendation(
                "Medium",
                f"Consider indexes or clustering on filter/join columns for {tbl}",
                "Useful if table is large and frequently queried on those keys.",
            )

    if db_type in ["PostgreSQL", "MySQL"]:
        add_recommendation(
            "Medium",
            "Review EXPLAIN / EXPLAIN ANALYZE plan",
            "Validate whether scans, joins, and sorts match expectations.",
        )
    elif db_type in ["Snowflake", "BigQuery"]:
        add_recommendation(
            "Medium",
            "Review partitioning / clustering / pruning effectiveness",
            "Cloud warehouses benefit heavily from scan reduction.",
        )
    elif db_type == "Teradata":
        add_recommendation(
            "Medium",
            "Review statistics and primary index suitability",
            "Teradata performance often depends on distribution and collected stats.",
        )

    summary_rows = [
        {"metric": "Detected Tables", "value": ", ".join(tables) if tables else "None detected"},
        {"metric": "Selected Columns", "value": selected_cols},
        {"metric": "Join Count", "value": join_count},
        {"metric": "Subquery Count", "value": subquery_count},
        {"metric": "Has WHERE", "value": "Yes" if has_where else "No"},
        {"metric": "Has GROUP BY", "value": "Yes" if has_group else "No"},
        {"metric": "Has ORDER BY", "value": "Yes" if has_order else "No"},
        {"metric": "Has LIMIT", "value": "Yes" if has_limit else "No"},
        {"metric": "Complexity Score (0-10)", "value": complexity_score},
        {"metric": "Analyzed At", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
    ]

    issues_df = pd.DataFrame(issues)
    recommendations_df = pd.DataFrame(recommendations)
    summary_df = pd.DataFrame(summary_rows)

    optimized_sql = build_optimized_example(cleaned, has_select_star, function_filters, db_type)

    return summary_df, issues_df, recommendations_df, optimized_sql


def build_optimized_example(query: str, has_select_star: bool, function_filters: list[str], db_type: str) -> str:
    optimized = query

    if has_select_star:
        optimized = re.sub(
            r"SELECT\s+\*",
            "SELECT\n    /* replace with required columns */\n    order_id,\n    customer_id,\n    order_date",
            optimized,
            flags=re.IGNORECASE,
        )

    optimized = re.sub(
        r"WHERE\s+YEAR\s*\(\s*([a-zA-Z0-9_.]+)\s*\)\s*=\s*(\d{4})",
        r"WHERE \1 >= '\2-01-01' AND \1 < '\2-12-31'",
        optimized,
        flags=re.IGNORECASE,
    )

    if "LIMIT" not in optimized.upper() and "GROUP BY" not in optimized.upper():
        optimized = optimized.rstrip().rstrip(";") + "\nLIMIT 100;"

    header = f"""-- Optimized example suggestion
-- Database Type: {db_type}
-- Notes:
-- 1. Select only required columns
-- 2. Use range predicates instead of function-wrapped filters
-- 3. Apply LIMIT for exploratory analysis
"""

    return header + "\n" + optimized


# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("🛠️ SQL Query Performance Analyzer")
st.caption(
    "A data engineer-style SQL review app for detecting anti-patterns, estimating complexity, and suggesting performance improvements."
)

st.markdown(
    """
### What this project demonstrates
- SQL anti-pattern detection
- query hygiene checks
- indexing and filtering suggestions
- warehouse optimization thinking
- production-style review workflow
"""
)

top_left, top_right = st.columns([2, 1])

with top_left:
    selected_sample = st.selectbox(
        "Load sample query",
        ["Custom Query"] + list(SAMPLE_QUERIES.keys()),
    )

with top_right:
    db_type = st.selectbox(
        "Database Type",
        ["PostgreSQL", "Snowflake", "BigQuery", "Teradata", "MySQL", "SQL Server"],
    )

default_query = ""
if selected_sample != "Custom Query":
    default_query = SAMPLE_QUERIES[selected_sample]

query = st.text_area(
    "Paste SQL Query",
    value=default_query,
    height=240,
)

col1, col2, col3 = st.columns(3)
with col1:
    table_size = st.selectbox("Estimated Table Size", ["Small", "Medium", "Large", "Very Large"])
with col2:
    st.metric("Query Length", len(query))
with col3:
    st.metric("Detected Keywords", len(re.findall(r"[A-Za-z_]+", query)))

run_analysis = st.button("Analyze Query", use_container_width=True)

if run_analysis:
    if not query.strip():
        st.warning("Please enter a SQL query.")
    else:
        summary_df, issues_df, recommendations_df, optimized_sql = analyze_sql(query, db_type, table_size)

        issue_count = len(issues_df)
        high_count = 0 if issues_df.empty else int((issues_df["severity"] == "High").sum())
        med_count = 0 if issues_df.empty else int((issues_df["severity"] == "Medium").sum())
        complexity_score = int(summary_df.loc[summary_df["metric"] == "Complexity Score (0-10)", "value"].iloc[0])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Issues Found", issue_count)
        k2.metric("High Severity", high_count)
        k3.metric("Medium Severity", med_count)
        k4.metric("Complexity Score", complexity_score)

        st.markdown("---")

        left, right = st.columns([1.1, 1])

        with left:
            st.subheader("Analysis Summary")
            render_html_table(summary_df, height=320)

        with right:
            st.subheader("Issue Severity Distribution")
            if issues_df.empty:
                st.success("No major issues detected.")
            else:
                severity_series = issues_df["severity"].value_counts()
                render_bar_chart(severity_series, "Issue Severity", "Severity", "Count")

        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Detected Issues", "Recommendations", "Optimized Example", "Engineering Notes"]
        )

        with tab1:
            st.subheader("Detected Issues")
            if issues_df.empty:
                st.success("No major anti-patterns detected.")
            else:
                render_html_table(issues_df, height=360)

        with tab2:
            st.subheader("Recommendations")
            render_html_table(recommendations_df, height=360)

        with tab3:
            st.subheader("Optimized Example Query")
            st.code(optimized_sql, language="sql")

        with tab4:
            st.subheader("Engineering Notes")
            st.markdown(
                """
**Checks included**
- SELECT * detection
- missing WHERE on large tables
- function-wrapped filter detection
- leading wildcard LIKE
- DISTINCT usage
- join complexity
- subquery detection
- ORDER BY / GROUP BY review
- UNION vs UNION ALL suggestion

**Typical data engineering use cases**
- query review before production deployment
- warehouse cost reduction
- scan reduction analysis
- BI/report query tuning
- developer enablement

**Best next upgrade ideas**
- EXPLAIN plan parser
- index recommendation engine
- query history upload
- estimated cost scoring by warehouse type
- before/after comparison mode
"""
            )