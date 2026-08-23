from flask import Blueprint, render_template, request
from statistics import mean, fmean, median
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

from pathlib import Path
name = Path(__file__).parent.name  # laab_report

from .bootstrap import PROFILES
from laab_inspector.inspector.profile import LAABProfile
from laab_inspector.inspector.ranking_utils import compute_partial_ranks


bp = Blueprint(
    name,
    __name__,
    template_folder="templates/",
    static_folder="static/",
    static_url_path=f"/static/{name}/",
)


INDEX_KEYS = [
    "name",
    "system",
    "lib",
    "version",
    "toolchain",
    "prec",
    "prob_size",
    "interface",
    "pinning",
    "options",
]

def get_features(step_specs):
    features = {}

    for key, value in step_specs.items():
        if key in INDEX_KEYS:
            continue

        if key == "flops":
            key = "#GFLOPS"
            value = f"{float(value) / 1e9:.5f}"

        features[key] = value

    return features


@bp.route("/")
def index():
    profiles = PROFILES.get_profiles()
    return render_template(f"{name}/index.html", profiles=profiles)


@bp.route("/compare", methods=["POST"])
def compare():
    profile_paths = request.form["profiles"].split(",")

    specs = {}
    profiles = []

    for path in profile_paths:
        profile = LAABProfile.load(PROFILES.profiles_root / path)
        _specs = profile.step_specs

        for k in INDEX_KEYS:
            specs.setdefault(k, set()).add(_specs[k])

        profiles.append(profile)

    attributes = []

    for k in list(specs):
        if len(specs[k]) > 1:
            attributes.append(k)
            specs.pop(k)

    for k in INDEX_KEYS:
        if k in specs:
            specs[k] = list(specs[k])[0]

    variants = {}
    records = []
    compute_perfs = {}
    exec_times = {}

    for i, profile in enumerate(profiles):
        variant = ""

        for attr in attributes:
            variant += f"/{profile.step_specs[attr]}"

        variants[variant] = profile_paths[i]

        for key in profile.compute_perfs.keys():
            _r = {"variant": variant}
            _r["key"] = key
            _r["npe"] = key.split("/")[0]
            _r["wd"] = key.split("/")[1]

            _compute_perfs = []
            # {npe/wd: {cb=id: [compute_perfs], ...}}
            for _, v in profile.compute_perfs[key].items():
                _compute_perfs.extend(v)
            compute_perfs.setdefault(key, {})[variant] = _compute_perfs
            _r["median_compute_perf"] = median(_compute_perfs)

            _exec_times = []
            for _, v in profile.exec_times[key].items():
                _exec_times.extend(v)
            _r["median_exec_time"] = median(_exec_times)
            _r["ncb"] = len(profile.exec_times[key])
            records.append(_r)

    compute_perfs = dict(sorted(compute_perfs.items(), key=lambda item: int(item[0].split("/")[0])))

    variant_ranks = {}
    for key in compute_perfs.keys():
        ranks_m1, ranks_m2, ranks_m3, nranks = compute_partial_ranks(compute_perfs[key], perf_indicator_key="GFLOPs/s")
        variant_ranks[key] = {
            "ranks_m1": ranks_m1,
            "ranks_m2": ranks_m2,
            "ranks_m3": ranks_m3,
            "nranks": nranks,
        }
        
    scaling_data = defaultdict(
        lambda: defaultdict(
            lambda: {
                "exec_time": [],
                "perf": [],
                "tag": [],
            }
        )
    )
    
    for r in records:
        x_value = r["npe"]
        tag = r["wd"]
        scaling_data[r["variant"]][x_value]["exec_time"].append(r["median_exec_time"])
        scaling_data[r["variant"]][x_value]["perf"].append(r["median_compute_perf"])
        scaling_data[r["variant"]][x_value]["tag"].append(tag)
        
    # Compute mean exec_time and perf for each variant and x_value.. and concat tags with '|'
    for variant, x_values in scaling_data.items():
        for x_value, values in x_values.items():
            values["exec_time"] = fmean(values["exec_time"])
            values["perf"] = fmean(values["perf"])
            values["tag"] = " | ".join(set(values["tag"]))
            
    #compute scaling efficiency for each variant and x_value
    for variant, x_values in scaling_data.items():
        reference_tasks = min(int(x) for x in x_values)
        reference_time = scaling_data[variant][str(reference_tasks)]["exec_time"]

        if reference_time <= 0:
            raise ValueError("Reference execution time must be greater than zero.")

        for x_value, values in x_values.items():
            tasks = int(x_value)
            exec_time = values["exec_time"]

            if exec_time <= 0:
                raise ValueError(f"Execution time for {tasks} tasks must be greater than zero.")

            values["efficiency"] = reference_time * reference_tasks / (exec_time * tasks) * 100.0
            

    return render_template(f"{name}/compare.html",
                           specs=specs,
                           variants=variants,
                           variant_records=records,
                           compute_perfs=compute_perfs,
                           variant_ranks=variant_ranks,
                           scaling_data=scaling_data)


@bp.route("/laab_report")
def laab_report():
    profile_path = request.args.get("profile_path")
    profile_path = PROFILES.profiles_root / profile_path

    if not profile_path.exists():
        msg = "Profile not found!"
        logger.error(f"404: {msg} - {profile_path}")
        return render_template(f"{name}/error.html", message=msg, code=404), 404

    try:
        profile = LAABProfile.load(profile_path)
    except Exception as e:
        msg = "Error loading profile."
        logger.error(f"500: {msg} - {e}")
        return render_template(f"{name}/error.html", message=msg, code=500), 500

    step_specs = profile.step_specs
    step_features = get_features(step_specs)
    
    run_specs = profile.run_specs
    run_records = []
    _cols = ["ts", "npe", "wd", "cb_id", "median_exec_time", "median_compute_perf"]
    run_feature_cols = []
    for key, r in run_specs.items():
        _r = {}
        _r["key"] = key
        _r["npe"] = key.split("/")[0]
        _r["wd"] = key.split("/")[1]
        _r["cb_id"] = key.split("/")[2]        
        _r["median_exec_time"] = profile.median_exec_time[key]
        _r["median_compute_perf"] = profile.median_compute_perf[key]
        for k,v in r.items():
            _r[k] = v
            if k not in _cols and k not in run_feature_cols:
                run_feature_cols.append(k)
        run_records.append(_r)
        
    # records = sorted(records, key=lambda item: int(item["key"].split("/")[0]))
    
    median_exec_time = profile.median_exec_time
    median_compute_perf = profile.median_compute_perf
    
    
    scaling_data = defaultdict(
        lambda: defaultdict(
            lambda: {
                "exec_time": [],
                "perf": [],
                "tag": [],
            }
        )    
    )
    for row in run_records:
        x_value = row["npe"]
        tag = row["wd"]
        scaling_data["measurement"][x_value]["exec_time"].append(row["median_exec_time"])
        scaling_data["measurement"][x_value]["perf"].append(row["median_compute_perf"])
        scaling_data["measurement"][x_value]["tag"].append(tag)
        
    for x_value, values in scaling_data["measurement"].items():
        values["exec_time"] = fmean(values["exec_time"])
        values["perf"] = fmean(values["perf"])
        values["tag"] = " | ".join(set(values["tag"]))

    reference_tasks = min(int(x) for x in scaling_data["measurement"])
    reference_time = scaling_data["measurement"][str(reference_tasks)]["exec_time"]        
    if reference_time <= 0:
        raise ValueError("Reference execution time must be greater than zero.")
    
    for x_value, values in scaling_data["measurement"].items():
        tasks = int(x_value)
        exec_time = values["exec_time"]

        if exec_time <= 0:
            raise ValueError(f"Execution time for {tasks} tasks must be greater than zero.")

        values["efficiency"] = reference_time * reference_tasks / (exec_time * tasks) * 100.0
            

    return render_template(f"{name}/laab_report.html",
                           name=step_specs["name"],
                           system=step_specs["system"],
                           lib=step_specs["lib"],
                           version=step_specs["version"],
                           toolchain=step_specs["toolchain"],
                           prec=step_specs["prec"],
                           prob_size=step_specs["prob_size"],
                           interface=step_specs["interface"],
                           pinning=step_specs["pinning"],
                           options=step_specs["options"],
                           step_features=step_features,
                           run_records=run_records,
                           run_feature_cols=run_feature_cols,
                           compute_perfs=profile.compute_perfs,
                           cb_ranks=profile.cb_ranks,
                           scaling_data=scaling_data,
                           profile_path=profile_path)


@bp.route("/host_info")
def host_info():
    profile_path = request.args.get("profile_path")
    exp_key = request.args.get("exp_key")
    profile_path = PROFILES.profiles_root / profile_path

    if not profile_path.exists():
        msg = "Profile not found!"
        logger.error(f"404: {msg} - {profile_path}")
        return render_template(f"{name}/error.html", message=msg, code=404), 404

    try:
        profile = LAABProfile.load(profile_path)
    except Exception as e:
        msg = "Error loading profile."
        logger.error(f"500: {msg} - {e}")
        return render_template(f"{name}/error.html", message=msg, code=500), 500

    if exp_key not in profile.host_specs:
        msg = "Host info not found!"
        logger.error(f"404: {msg} - {exp_key}")
        return render_template(f"{name}/error.html", message=msg, code=404), 404

    host_specs = profile.host_specs[exp_key]
    step_specs = profile.step_specs
    step_features = get_features(step_specs)

    run_specs = profile.run_specs[exp_key]
    median_exec_time = profile.median_exec_time[exp_key]
    median_compute_perf = profile.median_compute_perf[exp_key]

    return render_template(f"{name}/host_info.html",
                           **step_specs,
                           run_specs=run_specs,
                           step_features=step_features,
                           median_exec_time=median_exec_time,
                           median_compute_perf=median_compute_perf,
                           exp_key=exp_key,
                           host_specs=host_specs)

    
@bp.route('/acknowledgemnts')
def acknowledgements():
    return render_template(f'{name}/acknowledgements.html')