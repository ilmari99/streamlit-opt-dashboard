import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="Optimization Dashboard",
    page_icon="charts",
    layout="wide",
)

st.title("🧩 Optimization Process Dashboard")
st.write(f"Current Working Directory: {os.getcwd()}")

# Function to load data
@st.cache_data(ttl=5) # Cache for 5s
def load_data(experiment_dir):
    experiment_path = Path(experiment_dir)
    nodes_path = experiment_path / "nodes"
    lineage_path = experiment_path / "lineage.json"
    
    if not nodes_path.exists():
        return pd.DataFrame(), nx.DiGraph()

    # Load lineage
    lineage_data = {}
    if lineage_path.exists():
        try:
            with open(lineage_path, 'r') as f:
                lineage_data = json.load(f)
        except:
            pass

    # Load nodes
    nodes_data = []
    
    # Iterate through node directories
    for node_dir in nodes_path.iterdir():
        if not node_dir.is_dir():
            continue
        
        metadata_file = node_dir / "metadata.json"
        if not metadata_file.exists():
            continue
            
        try:
            with open(metadata_file, 'r') as f:
                meta = json.load(f)
                
            node_id = meta.get('id')
            parent_id = meta.get('parent_id')
            
            # Extract score
            score_data = meta.get('score', {})
            if isinstance(score_data, dict):
                score = score_data.get('total', 0.0)
            elif isinstance(score_data, (int, float)):
                score = float(score_data)
            else:
                score = 0.0
            
            # Status
            status = meta.get('status', 'UNKNOWN')
            if 'CORRECT' in status:
                color = 'green'
                status_short = 'Correct'
            elif 'INCORRECT' in status:
                color = 'red'
                status_short = 'Incorrect'
            else:
                color = 'gray'
                status_short = status.replace('NodeStatus.', '')

            # Features
            features = meta.get('feature_vector', [])
            
            # Additional Metrics Extraction
            # Check metadata first
            m_data = meta.get('metadata', {})
            # Check score breakdown runtime profile
            profile = meta.get('score', {}).get('breakdown', {}).get('runtime_profile', {})
            
            # Helper to get value from either source
            def get_val(key, default=None):
                val = profile.get(key)
                if val is None:
                    val = m_data.get(key)
                return val if val is not None else default

            wall_time = get_val('wall_time_s', 0.0)
            rss = get_val('max_rss_kb', 0.0)
            instructions = get_val('instructions', 0.0)
            
            nodes_data.append({
                'id': node_id,
                'parent_id': parent_id,
                'score': score,
                'status': status_short,
                'full_status': status,
                'color': color,
                'features': features,
                'created_at': pd.to_datetime(meta.get('created_at', 'now')),
                'code_summary': (
                    m_data.get('code_summary')
                    or m_data.get('llm_reasoning_content')
                    or m_data.get('llm_thinking_blocks')
                    or 'No summary'
                ),
                'wall_time': wall_time,
                'max_rss_kb': rss,
                'instructions': instructions
            })
                    
        except Exception as e:
            # st.error(f"Error loading node {node_dir.name}: {e}")
            pass

    df = pd.DataFrame(nodes_data)
    
    # Build Graph
    G = nx.DiGraph()
    if not df.empty:
        # Add nodes
        for _, row in df.iterrows():
            G.add_node(row['id'], **row.to_dict())
        
        # Add edges based on lineage.json if available, else parent_id
        edges = []
        if lineage_data:
            for parent, children in lineage_data.items():
                if parent == "root": # special key in lineage.json sometimes
                    continue 
                # Ensure parent is in graph (might be missing metadata?)
                # Actually we can add edges even if nodes are missing, seeing holes might be useful
                # But for visualization we prefer nodes to exist
                
                if isinstance(children, list):
                    for child in children:
                        if child in G.nodes and parent in G.nodes:
                             edges.append((parent, child))
                        elif child in G.nodes and parent == "root":
                             pass # Root handling if logical root
        
        # Fallback or supplement with parent_id from metadata if lineage didn't give edges
        if not edges: 
            for _, row in df.iterrows():
                if row['parent_id'] and row['parent_id'] != "root" and row['parent_id'] in G.nodes:
                    edges.append((row['parent_id'], row['id']))
        
        G.add_edges_from(edges)
        
    return df, G

def get_tree_layout(G):
    if len(G) == 0:
        return {}
        
    # Identification of roots
    roots = [n for n, d in G.in_degree() if d == 0]
    
    pos = {}
    # Simple layout strategy:
    # Y is depth. 
    # X is determined by post-order traversal to center parents over children.
    
    def get_depths(node, current_depth=0):
        depths = {node: current_depth}
        for child in G.successors(node):
            depths.update(get_depths(child, current_depth + 1))
        return depths

    all_depths = {}
    for root in roots:
        all_depths.update(get_depths(root))
        
    # Assign X coordinates
    # For each node, x is:
    # If leaf: counter++
    # If internal: average of children's x
    
    counter = [0]
    
    def assign_x(node):
        children = list(G.successors(node))
        if not children:
            x = counter[0]
            counter[0] += 1
            return x
        
        child_xs = [assign_x(child) for child in children]
        return sum(child_xs) / len(child_xs)
        
    x_coords = {}
    
    # Process each connected component/tree
    for root in roots:
        # Reset counter for each tree to avoid huge gaps? 
        # Actually better to keep accumulating to place trees side-by-side
        x_root = assign_x(root)
        x_coords[root] = x_root
        
        # We need to fill x_coords for all descendants
        # assign_x only returns for root, but we need to cache results.
        # Let's rewrite assign_x to populate dict
        pass

    # Re-running with population
    final_pos = {}
    current_x = [0]
    
    def process_place(node):
        children = list(G.successors(node))
        if not children:
            x = current_x[0]
            current_x[0] += 1.0 # Spacing
            final_pos[node] = (x, -all_depths.get(node, 0))
            return x
        
        child_xs = []
        for child in children:
            child_xs.append(process_place(child))
            
        x = sum(child_xs) / len(child_xs)
        final_pos[node] = (x, -all_depths.get(node, 0))
        return x

    for root in roots:
        process_place(root)
        current_x[0] += 1.0 # Gap between trees
        
    # Normalize positions
    # Check if any nodes missed (if cycles existed or disconnected parts not reachable from roots)
    for n in G.nodes():
        if n not in final_pos:
            final_pos[n] = (current_x[0], 0)
            current_x[0] += 1
            
    return final_pos

# Sidebar
st.sidebar.header("Experiment Settings")
experiments_root = Path("experiments")
if experiments_root.exists():
    experiments = [d.name for d in experiments_root.iterdir() if d.is_dir()]
    st.write(f"Found {len(experiments)} experiments: {experiments}")
    experiments = sorted(experiments)
    selected_experiment = st.sidebar.selectbox("Select Experiment", experiments)
else:
    st.error("No experiments folder found.")
    st.stop()

if selected_experiment:
    exp_path = experiments_root / selected_experiment
    
    # Load Data
    # with st.spinner(f"Loading data from {exp_path.resolve()}..."):
    df, G = load_data(str(exp_path.resolve()))
    
    st.write(f"Loaded {len(df)} nodes.")
    
    if df.empty:
        st.warning("Dataframe is empty. Checking nodes folder...")
        nodes_debug = exp_path / "nodes"
        if nodes_debug.exists():
            st.write(f"Nodes folder exists: {nodes_debug}")
            files = list(nodes_debug.iterdir())
            st.write(f"Found {len(files)} items in nodes folder.")
        else:
            st.error(f"Nodes folder not found at {nodes_debug}")
        st.stop()
        
    # --- Dashboard Layout ---
    
    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    best_score = df['score'].max()
    processed_nodes = len(df)
    correct_nodes = len(df[df['status'] == 'Correct'])
    best_node_row = df.loc[df['score'].idxmax()]
    best_node_id = best_node_row['id'][:8]
    
    c1.metric("Nodes Processed", processed_nodes)
    c2.metric("Best Score", f"{best_score:.4f}")
    c3.metric("Correct Solutions", correct_nodes)
    c4.metric("Best Node ID", best_node_id)
    
    # Graph & Leaderboard
    st.divider()
    
    # Color control
    color_metric = st.sidebar.selectbox(
        "Color Nodes By", 
        ["Status", "Score", "Wall Time", "Instructions", "Memory"]
    )
    
    col_graph, col_lb = st.columns([2, 1])
    
    with col_graph:
        st.subheader("Process Connectivity Graph")
        if len(G.nodes) > 0:
            # Layout
            try:
                # Use tree layout
                pos = get_tree_layout(G)
                if not pos:
                    k_val = 0.5 / np.sqrt(len(G.nodes) + 1)
                    pos = nx.spring_layout(G, seed=42, k=min(k_val, 2.0))
            except Exception as e:
                # st.warning(f"Sort layout failed: {e}")
                pos = nx.random_layout(G)
            
            edge_x = []
            edge_y = []
            for edge in G.edges():
                if edge[0] in pos and edge[1] in pos:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, 
                line=dict(width=0.5, color='#888'), 
                hoverinfo='none', 
                mode='lines'
            )

            # Node Coloring Logic
            node_x, node_y, node_text, node_colors, node_size = [], [], [], [], []
            
            # Helper to map metric to value
            metric_map = {
                "Score": "score",
                "Wall Time": "wall_time",
                "Instructions": "instructions",
                "Memory": "max_rss_kb"
            }
            
            val_list = []
            colorscale = 'RdYlGn'
            reversescale = False
            show_colorbar = False
            
            if color_metric != "Status":
                col_name = metric_map[color_metric]
                # Determine range for normalization if needed, or let plotly handle it
                show_colorbar = True
                if color_metric == "Score":
                    reversescale = False # High=Green
                else:
                    reversescale = True # Low=Green (Time, Memory, etc)
            
            for node in G.nodes():
                if node in pos:
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    n = G.nodes[node]
                    
                    status_lbl = n.get('status', '?')
                    score_fmt = n.get('score', 0)
                    
                    node_text.append(
                        f"ID: {n['id'][:8]}...<br>" + 
                        f"Score: {score_fmt:.4f}<br>" + 
                        f"Status: {status_lbl}<br>" +
                        f"Time: {n.get('wall_time',0):.4f}s"
                    )
                    
                    if color_metric == "Status":
                        node_colors.append(n.get('color', 'gray'))
                    else:
                        # For metric coloring, we only color CORRECT nodes by metric usually,
                        # but let's color everything and let user interpret.
                        val = n.get(metric_map[color_metric], 0)
                        node_colors.append(val)
                    
                    node_size.append(15 if n['score'] == best_score else 8)

            marker_dict = dict(
                 size=node_size, 
                 line_width=1, 
                 line_color='black'
            )
            
            if color_metric == "Status":
                marker_dict['color'] = node_colors
                marker_dict['showscale'] = False
            else:
                marker_dict['color'] = node_colors
                marker_dict['colorscale'] = colorscale
                marker_dict['reversescale'] = reversescale
                marker_dict['showscale'] = True
                marker_dict['colorbar'] = dict(title=color_metric)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=marker_dict
            )

            fig = go.Figure(data=[edge_trace, node_trace],
                           layout=go.Layout(
                               showlegend=False,
                               hovermode='closest',
                               margin=dict(b=20,l=5,r=5,t=40),
                               xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                               height=500,
                               paper_bgcolor='rgba(0,0,0,0)',
                               plot_bgcolor='rgba(0,0,0,0)'
                           ))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No graph data")

    with col_lb:
        st.subheader("Leaderboard")
        
        # Prepare leaderboard columns
        lb_df = df.copy()
        
        # Reorder columns: score, status, metrics, id
        cols = ['score', 'status', 'wall_time', 'instructions', 'max_rss_kb', 'id']
        lb_cols = [c for c in cols if c in lb_df.columns]
        
        st.dataframe(
            lb_df[lb_cols].sort_values('score', ascending=False),
            use_container_width=True,
            height=500,
            column_config={
                "score": st.column_config.NumberColumn("Score", format="%.4f"),
                "wall_time": st.column_config.NumberColumn("Time (s)", format="%.4f"),
                "instructions": st.column_config.NumberColumn("Instr.", format="%.2e"),
                "max_rss_kb": st.column_config.NumberColumn("Mem (KB)"),
                "id": st.column_config.TextColumn("ID", width="small"),
            },
            hide_index=True 
        )

    # Score Evolution
    st.divider()
    st.subheader("Score Evolution")
    
    if not df.empty and 'created_at' in df.columns:
        # Sort by date
        df_sorted = df.sort_values('created_at')
        
        # Create scatter plot
        fig_evol = px.scatter(
            df_sorted, 
            x='created_at', 
            y='score',
            color='status',
            color_discrete_map={'Correct': 'green', 'Incorrect': 'red', 'UNKNOWN': 'gray'},
            hover_data=['id'],
            title="Score History"
        )
        
        st.plotly_chart(fig_evol, use_container_width=True)
        
    # Analysis Section
    st.divider()
    st.subheader("Analysis & Distributions")
    
    a1, a2 = st.columns(2)
    
    with a1:
        st.markdown("**Feature Vector Space**")
        method = st.radio("Projection Method", ["PCA", "t-SNE"], horizontal=True)
        
        # Filter nodes with features
        feat_df = df[df['features'].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
        # Ensure score is numeric for plotting sizes
        feat_df['score_numeric'] = pd.to_numeric(feat_df['score'], errors='coerce').fillna(0)
        
        if len(feat_df) >= 3:
            try:
                features_matrix = np.array(feat_df['features'].tolist())
                # Handle potential NaNs in features
                features_matrix = np.nan_to_num(features_matrix)
                
                if method == "PCA":
                    pca = PCA(n_components=2)
                    comps = pca.fit_transform(features_matrix)
                    title = f"PCA of {len(features_matrix[0])} Features"
                else: # t-SNE
                    n_samples = len(features_matrix)
                    # Perplexity must be less than n_samples
                    perp = min(30, max(5, int(n_samples / 4))) 
                    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='random', learning_rate='auto')
                    comps = tsne.fit_transform(features_matrix)
                    title = f"t-SNE (perp={perp})"
                
                feat_df['C1'] = comps[:, 0]
                feat_df['C2'] = comps[:, 1]

                # Ensure positive values for size to avoid Plotly errors
                # We use a base size of 5 and add the max of 0 and the score
                feat_df['plot_size'] = np.maximum(feat_df['score_numeric'], 0) + 5
                
                fig_dim = px.scatter(
                    feat_df, x='C1', y='C2',
                    color=metric_map[color_metric] if color_metric != "Status" else 'status',
                    size='plot_size',
                    hover_data=['id', 'score', 'wall_time', 'instructions'],
                    color_discrete_map={'Correct': 'green', 'Incorrect': 'red', 'UNKNOWN': 'gray'} if color_metric == "Status" else None,
                    color_continuous_scale='RdYlGn' if color_metric == "Score" else ('RdYlGn_r' if color_metric != "Status" else None),
                    title=title
                )
                st.plotly_chart(fig_dim, use_container_width=True)
            except Exception as e:
                st.warning(f"Error computing projection: {e}")
        else:
            st.info("Not enough data points for projection (need at least 3).")

    with a2:
        st.markdown("**Score Distribution**")
        fig_hist = px.histogram(df, x="score", color="status", nbins=20, 
                                color_discrete_map={'Correct': 'green', 'Incorrect': 'red', 'UNKNOWN': 'gray'},
                                title="Histogram of Scores")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Auto-refresh logic (placed at the end to ensure rendering completes)
    if st.sidebar.checkbox("Auto Refresh", value=True):
        time.sleep(2)
        st.rerun()
