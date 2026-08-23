from partial_ranker import QuantileComparer, PartialRanker, Method
import numpy as np


def compute_partial_ranks(measurements,q_max=75, q_min=25, remove_outliers=False, higher_is_better=True, perf_indicator_key="avg_perf_indicator"):
    
    ranking_data = measurements.copy()
    if higher_is_better:
        ranking_data = {k: -1*np.array(v) for k, v in ranking_data.items()} 
    
    cm = QuantileComparer(ranking_data)
    cm.compute_quantiles(q_max=q_max, q_min=q_min, outliers=remove_outliers)
    cm.compare()
    pr = PartialRanker(cm)
    
    pr.compute_ranks(Method.DFG)
    nranks_m1 = len(pr.get_ranks())
    ranks_m1 = pr.ranker._obj_rank
    
    pr.compute_ranks(Method.DFGReduced)
    nranks_m2 = len(pr.get_ranks())
    ranks_m2 = pr.ranker._obj_rank
    
    pr.compute_ranks(Method.Min)
    nranks_m3 = len(pr.get_ranks())
    ranks_m3 = pr.ranker._obj_rank
    
    nranks = f'{nranks_m1}-{nranks_m2}-{nranks_m3}'
    
    
    return ranks_m1, ranks_m2, ranks_m3, nranks    