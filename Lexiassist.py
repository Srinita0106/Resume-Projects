import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import graphviz
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sql_metadata import Parser

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Semantic SQL Search", layout="wide")

# Initialize local sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Database connection
def connect_to_db(db_name=None):
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root2003",
        database=db_name
    )

# Get list of available databases
@st.cache_data
def get_available_databases():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root2003"
        )
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall() if db[0] not in ('information_schema', 'mysql', 'performance_schema', 'sys')]
        cursor.close()
        conn.close()
        return databases
    except Exception as e:
        st.error(f"Error connecting to MySQL: {str(e)}")
        return []

# Schema understanding
@st.cache_data(ttl=3600)
def get_schema_info(db_name):
    conn = connect_to_db(db_name)
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SHOW TABLES")
    tables = [list(table.values())[0] for table in cursor.fetchall()]
    
    schema_info = {"tables": {}, "relationships": [], "database": db_name}
    
    for table in tables:
        cursor.execute(f"DESCRIBE {table}")
        columns = cursor.fetchall()
        
        cursor.execute(f"SELECT * FROM {table} LIMIT 1")
        sample = cursor.fetchone()
        
        schema_info["tables"][table] = {
            "columns": [col['Field'] for col in columns],
            "types": {col['Field']: col['Type'] for col in columns},
            "sample": sample
        }
    
    # Get foreign keys
    cursor.execute("""
        SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE REFERENCED_TABLE_NAME IS NOT NULL
        AND TABLE_SCHEMA = DATABASE()
    """)
    schema_info["relationships"] = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return schema_info

# Create schema embeddings
@st.cache_data
def create_schema_embeddings(schema_info):
    embeddings = {}
    for table, info in schema_info["tables"].items():
        # Embed table name and description
        table_desc = f"{table} table with columns: {', '.join(info['columns'])}"
        table_embedding = model.encode(table_desc)
        
        # Embed column names with their types
        column_embeddings = {}
        for column in info['columns']:
            col_desc = f"{column} ({info['types'][column]})"
            column_embeddings[column] = model.encode(col_desc)
        
        embeddings[table] = {
            "table_embedding": table_embedding,
            "columns": column_embeddings
        }
    return embeddings

# Find closest schema matches
def find_schema_matches(query, schema_info, schema_embeddings):
    query_embedding = model.encode(query)
    
    # Find closest table
    table_scores = []
    for table, embeddings in schema_embeddings.items():
        similarity = np.dot(query_embedding, embeddings["table_embedding"])
        table_scores.append((table, similarity))
    
    # Sort by similarity score
    table_scores.sort(key=lambda x: x[1], reverse=True)
    best_table = table_scores[0][0]
    
    # Find closest columns
    column_scores = []
    for column, embedding in schema_embeddings[best_table]["columns"].items():
        similarity = np.dot(query_embedding, embedding)
        column_scores.append((column, similarity))
    
    column_scores.sort(key=lambda x: x[1], reverse=True)
    
    return best_table, [col[0] for col in column_scores[:5]]  # return top 5 columns

# Rule-based SQL generator
def generate_sql(query, schema_info, schema_embeddings):
    table, columns = find_schema_matches(query, schema_info, schema_embeddings)
    
    # Basic query construction
    select_columns = ", ".join(columns)
    
    # Simple condition detection
    conditions = []
    
    # Time-related conditions
    time_phrases = {
        "last week": ">= DATE_SUB(CURDATE(), INTERVAL 1 WEEK)",
        "last month": ">= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)",
        "last year": ">= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)",
        "today": "= CURDATE()"
    }
    
    date_column = None
    for col in columns:
        if "date" in col.lower() or "time" in col.lower():
            date_column = col
            break
    
    for phrase, sql_phrase in time_phrases.items():
        if phrase in query.lower() and date_column:
            conditions.append(f"{date_column} {sql_phrase}")
    
    # Negation detection
    negation_words = ["not", "didn't", "haven't", "hasn't", "no"]
    for word in negation_words:
        if word in query.lower():
            # Find the verb after negation
            query_lower = query.lower()
            neg_index = query_lower.find(word)
            next_word = query_lower[neg_index:].split()[1] if len(query_lower[neg_index:].split()) > 1 else ""
            
            # Map to possible column names
            for col in columns:
                if next_word in col.lower():
                    conditions.append(f"{col} IS NULL OR {col} = 0")
                    break
    
    # Basic number filters
    number_words = ["more than", "greater than", "over", "less than", "under"]
    for word in number_words:
        if word in query.lower():
            # Try to extract the number
            parts = query.lower().split(word)
            if len(parts) > 1:
                num_part = parts[1].strip().split()[0]
                if num_part.replace('.', '').isdigit():
                    for col in columns:
                        if "amount" in col.lower() or "price" in col.lower() or "total" in col.lower():
                            operator = ">" if word in ["more than", "greater than", "over"] else "<"
                            conditions.append(f"{col} {operator} {num_part}")
                            break
    
    # Build WHERE clause
    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)
    
    # Basic SQL generation
    sql = f"SELECT {select_columns} FROM {table} {where_clause} LIMIT 100"
    
    return sql

# Create schema diagram
def create_schema_diagram(schema_info):
    graph = graphviz.Digraph()
    
    # Add tables as nodes
    for table_name, table_info in schema_info["tables"].items():
        columns = "\l".join([f"{col} ({table_info['types'][col]})" for col in table_info["columns"]])
        graph.node(table_name, f"{table_name}\l{columns}", shape="box")
    
    # Add relationships as edges
    for rel in schema_info["relationships"]:
        graph.edge(
            rel["TABLE_NAME"], 
            rel["REFERENCED_TABLE_NAME"],
            label=f"{rel['COLUMN_NAME']} → {rel['REFERENCED_COLUMN_NAME']}"
        )
    
    return graph

# Create ER diagram
def create_er_diagram(schema_info):
    graph = graphviz.Digraph(engine='neato', graph_attr={'splines': 'ortho'})
    
    # Add tables as nodes
    for table_name, table_info in schema_info["tables"].items():
        # Identify primary keys
        pk_columns = [col for col in table_info["columns"] 
                     if "PRIMARY" in str(table_info["types"][col]).upper()]
        
        # Format columns with PK indicators
        columns = []
        for col in table_info["columns"]:
            col_str = f"*{col}*" if col in pk_columns else col
            columns.append(f"{col_str} ({table_info['types'][col]})")
        
        columns_str = "\l".join(columns)
        graph.node(table_name, f"<<b>{table_name}</b>>\l{columns_str}", shape="none")
    
    # Add relationships as edges
    for rel in schema_info["relationships"]:
        graph.edge(
            rel["TABLE_NAME"], 
            rel["REFERENCED_TABLE_NAME"],
            label=f"{rel['COLUMN_NAME']} → {rel['REFERENCED_COLUMN_NAME']}",
            dir="both",
            arrowtail="crow",
            arrowhead="none"
        )
    
    return graph

# Generate visualizations based on query results
def generate_visualizations(df, sql_query):
    try:
        # Parse SQL to understand what was selected
        parsed = Parser(sql_query)
        selected_columns = parsed.columns_dict.get("select", [])
        
        # Remove table prefixes if they exist
        clean_columns = [col.split('.')[-1].strip('`') for col in selected_columns]
        
        # Create tabs for different visualization types
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📈 Line Chart", "🍩 Pie Chart", "📋 Data Table"])
        
        with tab1:
            st.subheader("Data Summary")
            st.write(df.describe(include='all'))
            
            # Auto-detect numeric columns for histogram
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                selected_num_col = st.selectbox("Select numeric column for histogram", numeric_cols)
                fig = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Line Chart")
            # Try to find date/time columns
            date_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['date', 'time', 'year', 'month', 'day'])]
            
            if len(date_cols) > 0:
                date_col = st.selectbox("Select date column", date_cols)
                value_col = st.selectbox("Select value column", [col for col in df.columns if col != date_col])
                
                if st.button("Generate Line Chart"):
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        fig = px.line(df.sort_values(date_col), x=date_col, y=value_col, 
                                      title=f"{value_col} over Time")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Couldn't create line chart: {str(e)}")
            else:
                st.warning("No date/time columns found for line chart")
        
        with tab3:
            st.subheader("Pie Chart")
            # Try to find categorical columns
            cat_cols = [col for col in df.columns if len(df[col].unique()) <= 20 and len(df[col].unique()) > 1]
            
            if len(cat_cols) > 0:
                cat_col = st.selectbox("Select categorical column", cat_cols)
                
                if st.button("Generate Pie Chart"):
                    fig = px.pie(df, names=cat_col, title=f"Distribution of {cat_col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No suitable categorical columns found for pie chart")
        
        with tab4:
            st.subheader("Detailed Data View")
            st.dataframe(df)
    
    except Exception as e:
        st.error(f"Error generating visualizations: {str(e)}")

# Main app function
def main():
    st.title("🔍 Enhanced Semantic SQL Search Engine")
    st.write("Ask your database questions in plain English - no API keys required")
    
    # Database selection
    available_dbs = get_available_databases()
    selected_db = st.selectbox("Select Database", available_dbs)
    
    if not selected_db:
        st.warning("Please select a database to continue")
        return
    
    # Initialize session state
    if 'schema_info' not in st.session_state or st.session_state.schema_info.get("database") != selected_db:
        with st.spinner(f"Loading {selected_db} schema..."):
            st.session_state.schema_info = get_schema_info(selected_db)
            st.session_state.schema_embeddings = create_schema_embeddings(st.session_state.schema_info)
    
    # Schema visualization tabs
    tab1, tab2, tab3 = st.tabs(["🗃️ Schema Browser", "📐 Schema Diagram", "🔗 ER Diagram"])
    
    with tab1:
        # Sidebar with schema info
        with st.sidebar:
            st.subheader("Database Schema")
            selected_table = st.selectbox("Select table", list(st.session_state.schema_info["tables"].keys()))
            
            if selected_table:
                st.write(f"**Columns in {selected_table}:**")
                for col in st.session_state.schema_info["tables"][selected_table]["columns"]:
                    st.write(f"- {col} ({st.session_state.schema_info['tables'][selected_table]['types'][col]})")
                
                st.write("**Sample row:**")
                st.json(st.session_state.schema_info["tables"][selected_table]["sample"])
    
    with tab2:
        st.subheader("Database Schema Diagram")
        schema_graph = create_schema_diagram(st.session_state.schema_info)
        st.graphviz_chart(schema_graph)
    
    with tab3:
        st.subheader("Entity Relationship Diagram")
        er_graph = create_er_diagram(st.session_state.schema_info)
        st.graphviz_chart(er_graph)
    
    # Main query interface
    st.divider()
    query = st.text_input(
        "Ask your database question:", 
        placeholder="e.g., Show me customers who bought in the last month but didn't open our emails",
        key="query_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Generate SQL", type="primary"):
            if query:
                with st.spinner("Generating SQL query..."):
                    try:
                        sql_query = generate_sql(query, st.session_state.schema_info, st.session_state.schema_embeddings)
                        st.session_state.generated_sql = sql_query
                    except Exception as e:
                        st.error(f"Error generating SQL: {str(e)}")
            else:
                st.warning("Please enter a question")
    
    with col2:
        if st.button("Clear"):
            st.session_state.pop("generated_sql", None)
            st.experimental_rerun()
    
    if 'generated_sql' in st.session_state:
        st.subheader("Generated SQL")
        st.code(st.session_state.generated_sql, language="sql")
        
        if st.button("Execute Query"):
            try:
                conn = connect_to_db(selected_db)
                df = pd.read_sql(st.session_state.generated_sql, conn)
                
                st.subheader(f"Results ({len(df)} rows)")
                
                # Generate visualizations
                generate_visualizations(df, st.session_state.generated_sql)
                
                # Export options
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download as CSV",
                    data=csv,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Error executing query: {str(e)}")
            finally:
                if 'conn' in locals():
                    conn.close()

if __name__ == "__main__":
    main()