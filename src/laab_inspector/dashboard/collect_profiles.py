import argparse
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from ..inspector.profile import LAABProfile


def laab_collect_profiles():
    parser = argparse.ArgumentParser(description="Prepare dashboard data")
    parser.add_argument("profiles_root", help="Path to the profiles root directory.")
    args = parser.parse_args()
    
    profiles_root = Path(args.profiles_root).expanduser().resolve()
    profile_paths = [p.resolve() for p in profiles_root.rglob("*.laab") if p.is_file()]

    profiles = []
    for path in profile_paths:
        p = LAABProfile.load(path)
        step_specs = p.step_specs
        step_specs["profile_path"] = str(path.relative_to(profiles_root))
        artifact_path = path.parent.parent / "artifact.tar.gz"
        step_specs["artifact_path"] = str(artifact_path.relative_to(profiles_root))
        profiles.append(step_specs)
        
    df = pd.DataFrame(profiles)
    profiles = df.sort_values(["name", "system", "lib", "version", "toolchain", "prec"]).reset_index(drop=True).to_dict(orient="records")
    
    old_files  = list(profiles_root.glob("profiles.*.json"))
            
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = profiles_root / f"profiles.{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=4, default=str)
        
    for old_file in old_files:
        old_file.unlink()

    print(f"Wrote {len(profiles)} profiles to {output_path}")
    
