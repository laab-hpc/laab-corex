from dataclasses import dataclass
from pathlib import Path
import glob
from collections import defaultdict

#INPUT

# [LAAB-STEP] cblas/dgemm | dt=... | prob_size=(A=3000x3000, B=3000x3000) | prec="float64" | flops=54000000000 | interface=cblas
# [LAAB-RUN] cblas/dgemm | np=4 | execution=OMP | l2_norm=..
# [LAAB-HOST] cblas/dgemm | hostname=b-cn1512.hpc2n.umu.se | core_id=18 
# [LAAB] cblas/dgemm | rep=0 | dt=2026-06-07 18:18:44 | dur (s)= 0.06155 | perf (GFLOP/s)= 877.31329  
# [LAAB] cblas/dgemm | rep=1 | dt=2026-06-07 18:18:44 | dur=0.06135 s | perf=880.24424 GFLOP/s

#OUTPUT
# step_specs
# {
#     "xgemm/A=3000x3000+B=3000x3000/float64": { "dt":..., "flops":54000000000, "interface": "cblas" },
# }

# host_specs
# {
#     "xgemm/A=3000x3000+B=3000x3000/float64": [
#         { "hostname":..., "core_id":18 }, 
#     ]
# }

# run_specs
# {
#     "xgemm/A=3000x3000+B=3000x3000/float64": { "np":4, "execution": "OMP" },
# }

# benchmarks
# {
#     "xgemm/A=3000x3000+B=3000x3000/float64": [
#         { "rep":0, "dt":..., "dur":0.06155, "perf":877.31329 },
#         { "rep":1, "dt":..., "dur":0.06135, "perf":880.24424 },
#     ]
# } 


@dataclass
class TracePayload:
    step_specs: dict
    host_specs: dict
    run_specs: dict
    benchmarks: dict
    
    @classmethod
    def from_traces(cls, trace_dir: Path):
        trace_files = cls._find_trace_files(trace_dir)
        if not trace_files:
            raise ValueError(f"No .log files found in {trace_dir}")
        step_specs, host_specs, run_specs, benchmarks = cls._parse_trace_files(trace_files)
    
        return cls(step_specs, host_specs, run_specs, benchmarks)
    
    @staticmethod
    def _find_trace_files(trace_dir: Path) -> list[Path]:
        log_files = glob.glob(str(trace_dir / "traces.*.log"))
        return log_files
    
    @staticmethod
    def _parse_trace_files(trace_files: list[Path]) -> tuple[dict, dict, dict]:
        step_specs = defaultdict(dict)
        host_specs = defaultdict(list)
        run_specs = defaultdict(dict)
        benchmarks = defaultdict(list)
        key = ""
        for trace_file in trace_files:
            with open(trace_file, "r") as f:
                for line in f:
                    if line.startswith("[LAAB-STEP]"):
                        # This entry should be available in all files before all other entries.
                        record = TracePayload._parse_record(line)
                        key = f"{record['interface']}/{record['prob_size']}/{record['prec']}"
                        # del record['prob_size']
                        # del record['prec']
                        step_specs[key] = record
                    elif line.startswith("[LAAB-HOST]"):
                        record = TracePayload._parse_record(line)
                        host_specs[key].append(record)
                    elif line.startswith("[LAAB-RUN]"):
                        record = TracePayload._parse_record(line)
                        run_specs[key] = record
                    elif line.startswith("[LAAB]"):
                        record = TracePayload._parse_record(line)
                        benchmarks[key].append(record)
                        
        return step_specs, host_specs, run_specs, benchmarks
                            
                        
    @staticmethod
    def _parse_record(line: str):
        record = {}
        for part in line.strip().split("|")[1:]:
            key, value = part.strip().split("=", 1)
            record[key.strip()] = value.strip().strip('"')
        return record
                        
