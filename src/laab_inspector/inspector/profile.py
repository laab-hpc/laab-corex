from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
from statistics import mean, median
from pathlib import Path
import pickle
import fcntl
from .trace_payload import TracePayload as Payload
from .ranking_utils import compute_partial_ranks

@dataclass
class LAABProfile:
    step_specs: dict = field(default_factory=dict)
    host_specs: dict = field(default_factory=dict)
    run_specs: dict = field(default_factory=dict)

    exec_times: dict = field(default_factory=dict)
    mean_exec_time: dict = field(default_factory=dict)
    median_exec_time: dict = field(default_factory=dict)

    compute_perfs: dict = field(default_factory=dict)
    mean_compute_perf: dict = field(default_factory=dict)
    median_compute_perf: dict = field(default_factory=dict)

    # scalability: dict = field(default_factory=dict)
    cb_ranks: dict = field(default_factory=dict)
    
    def save(self, path: Path):
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            raise RuntimeError(f"Error saving profile: {e}") from e
            
            
    @classmethod
    def load(cls, path: Path):
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading profile: {e}") from e
        return obj
    
    

def add_profile_entry(trace_dir, profile_dir, name, system, lib, version, toolchain, np, work_dist, pinning, cb_id, options=""):
    payload = Payload.from_traces(trace_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    for key, benchmarks in payload.benchmarks.items():
                  
        step_specs = {
            "name": name,
            "system": system,
            "lib": lib,
            "version": version,
            "toolchain": toolchain,
            "pinning": pinning,
            "options": options,
            **payload.step_specs[key]
        }
        
        interface, prob_size, prec = key.split("/")
        profile_path = profile_dir / f"{interface}_{prob_size}_{prec}_{pinning}.laab"
        profile_path_lock = profile_path.parent / ".locks" / f"{profile_path.name}.lock"
        profile_path_lock.parent.mkdir(parents=True, exist_ok=True)
        profile_path_lock = open(profile_path_lock, "w")
        fcntl.lockf(profile_path_lock, fcntl.LOCK_EX)
        
        if profile_path.exists():
            profile = LAABProfile.load(profile_path)
            if profile.step_specs != step_specs:
                raise ValueError(f"Profile {profile_path} already exists with different step specs")
        else:
            profile = LAABProfile()
            profile.step_specs = step_specs
            

        
        entry_key = f"{np}/{work_dist}/{cb_id}"

        df = pd.DataFrame(benchmarks)  
        
        exec_times = df["exec_time (s)"].astype(float).tolist()
        mean_exec_time = mean(exec_times)
        median_exec_time = median(exec_times)
        
        # profile.exec_times[entry_key]  = exec_times
        profile.exec_times.setdefault(f"{np}/{work_dist}", {})[f"cb={cb_id}"] = exec_times
        profile.mean_exec_time[entry_key] = mean_exec_time
        profile.median_exec_time[entry_key] = median_exec_time

        compute_perfs = df["perf (GFLOP/s)"].astype(float).tolist()
        mean_compute_perf = mean(compute_perfs)
        median_compute_perf = median(compute_perfs)
        
        # profile.compute_perfs[f"{np}/{work_dist}"][f"{cb_id}"] = compute_perfs
        profile.compute_perfs.setdefault(f"{np}/{work_dist}", {})[f"cb={cb_id}"] = compute_perfs
        profile.mean_compute_perf[entry_key]  = mean_compute_perf
        profile.median_compute_perf[entry_key]  = median_compute_perf
        

        profile = update_cb_rank_metric(profile, f"{np}/{work_dist}")
        
                
        profile.host_specs[entry_key] = payload.host_specs[key]
        profile.run_specs[entry_key] = payload.run_specs[key]
        
        profile.save(profile_path)
        
        fcntl.lockf(profile_path_lock, fcntl.LOCK_UN)
        profile_path_lock.close()
        
    
def update_cb_rank_metric(profile, key):
        
    ranks_m1, ranks_m2, ranks_m3, nranks = compute_partial_ranks(profile.compute_perfs[key], perf_indicator_key="GFLOPs/s")
    
    profile.cb_ranks[key] = {
        "ranks_m1": ranks_m1,
        "ranks_m2": ranks_m2,
        "ranks_m3": ranks_m3,
        "nranks": nranks
    }
    
    return profile
        
                     
def delete_profile_entry(interface, prob_size, prec, np, work_dist, pinning, cb_id, profile_dir):
    profile_path = profile_dir / f"{interface}_{prob_size}_{prec}_{pinning}.laab"
    
    if not profile_path.exists():
        raise ValueError(f"Profile {profile_path} does not exist")
    
    profile_path_lock = profile_path.parent / ".locks" / f"{profile_path.name}.lock"
    profile_path_lock.parent.mkdir(parents=True, exist_ok=True)
    profile_path_lock = open(profile_path_lock, "w")
    fcntl.lockf(profile_path_lock, fcntl.LOCK_EX)
    
    profile = LAABProfile.load(profile_path)
    entry_key = f"{np}/{work_dist}/{cb_id}"

    profile.exec_times.get(f"{np}/{work_dist}", {}).pop(f"{cb_id}", None)
    if profile.exec_times[f"{np}/{work_dist}"] == {}:
        del profile.exec_times[f"{np}/{work_dist}"]
    profile.mean_exec_time.pop(entry_key, None)
    
    profile.compute_perfs.get(f"{np}/{work_dist}", {}).pop(f"{cb_id}", None)
    if profile.compute_perfs[f"{np}/{work_dist}"] == {}:
        del profile.compute_perfs[f"{np}/{work_dist}"]
        
    profile.mean_compute_perf.pop(entry_key, None)
    profile.cb_ranks.pop(entry_key, None)
    profile.host_specs.pop(entry_key, None)
    profile.run_specs.pop(entry_key, None)
    profile.host_specs.pop(entry_key, None)
        
    profile.save(profile_path)
    
    fcntl.lockf(profile_path_lock, fcntl.LOCK_UN)
    profile_path_lock.close()
        

        
    
    