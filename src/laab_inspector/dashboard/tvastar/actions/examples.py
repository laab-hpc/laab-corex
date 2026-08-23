import pandas as pd
import pickle
from types import SimpleNamespace
import os

def get_simple_df():
    # Column schema: (id, display_name)
    col_schema = [
        {"id": "name", "label": "Name", "type": "string"},
        {"id": "age", "label": "Age", "type": "integer"},
        {"id": "city", "label": "City", "type": "string"},
        {"id": "score", "label": "Score", "type": "decimal"},
    ]

    # DataFrame uses IDs, not labels
    df = pd.DataFrame([
        {"name": "Alice",   "age": 30, "city": "New York",    "score": 85.0243834},
        {"name": "Bob",     "age": 25, "city": "Los Angeles", "score": 90.08234},
        {"name": "Charlie", "age": 35, "city": "Chicago",     "score": 78.762342},
        {"name": "Diana",   "age": 28, "city": "Miami",       "score": 92.927324},
        {"name": "Ethan",   "age": 32, "city": "Seattle",     "score": 88.27364},
    ])
    
    pk = "name"
    
    return col_schema, pk, df

def get_ranked_bp_data():
    # Sample boxplot data for different models
    bp_data = {
        "Model_A": [0.75, 0.80, 0.78, 0.82, 0.77],
        "Model_B": [0.65, 0.70, 0.68, 0.72, 0.66],
        "Model_C": [0.85, 0.88, 0.87, 0.90, 0.86],
    }
    
    # Ranks for the models
    ranks = {
        "Model_A": 1,
        "Model_B": 2,
        "Model_C": 0,
    }
    
    
    return bp_data, ranks

def get_dfg_data(data_dir):
    # read svg from path eg_data/dfg_ranks.svg
    dfg_ranks_svg = os.path.join(data_dir, "dfg_ranks.svg") 
    with open(dfg_ranks_svg, "r") as f:
        dfg_svg = f.read()
    
    dfg_context = os.path.join(data_dir, "dfg_ranks_context.pkl")   
    with open(dfg_context, "rb") as f:
        context = pickle.load(f)
        context = SimpleNamespace(**context)
    
    node_records = {}
    for activity in context.activities:
        records = []
        for obj, rank in context.activity_obj_pclass[activity].items():
            records.append({
                "rank": rank,
                "obj": obj,
                "niter": len(context.activity_obj_sbw_list[activity][obj]),
                "io_mean": context.activity_obj_io_mb_mean[activity][obj],
                "sbw_mean": context.activity_obj_sbw_mean[activity][obj],
                "a_counts": context.activity_obj_a_counts_mean[activity][obj],
            })
        node_records[activity] = records
  
    col_schema = [
        {"id": "rank", "label": "Node Rank", "type": "integer"},
        {"id": "obj", "label": "Command", "type": "string"},
        {"id": "niter", "label": "Num iters", "type": "integer"},
        {"id": "io_mean", "label": "Avg I/O (MB)", "type": "decimal"},
        {"id": "sbw_mean", "label": "Avg SBW (MB/s)", "type": "decimal"},
        {"id": "a_counts", "label": "Avg Access Counts", "type": "decimal"},
    ]
                    
    node_bp_data = context.activity_obj_sbw_list
    node_ranks = context.activity_obj_pclass       
                   
        
    node_info = {
        "nodes": context.activities,
        "col_schema": col_schema,
        "node_records": node_records,
        "pk": "obj",
        "bp_data": context.activity_obj_sbw_list,
        "ranks": context.activity_obj_pclass,
        "bp_title": "sBW Boxplot",
        "bp_x_label": "sBW (MB/s)",
    }
    
    return dfg_svg, node_info