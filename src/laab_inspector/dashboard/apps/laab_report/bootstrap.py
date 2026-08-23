import logging
logger = logging.getLogger(__name__)
import os
import json
from pathlib import Path

class ProfilesIndex:
    def __init__(self, profiles_root):
        self.profiles_root = Path(profiles_root).expanduser()
        self.profiles_file = None
        self.profiles = []
        self.version = None
        
        self._load_profiles()
    
    def _load_profiles(self):
        self.profiles_file = next(self.profiles_root.glob("profiles.*.json"), None)
        
        if self.profiles_file is None:
            logger.error(f"Could not find profiles.*.json file in {self.profiles_root}")
            raise FileNotFoundError(f"Could not find profiles.*.json file in {self.profiles_root}")
        
        self.version = self.profiles_file.name.split(".")[1]
        with open(self.profiles_file, "r", encoding="utf-8") as f:
            self.profiles = json.load(f)
        logger.info(f"[IO] Loaded {len(self.profiles)} profiles from {self.profiles_file}")
                      
    def get_profiles(self):
        if self.profiles_file.exists():
            return self.profiles
        else:
            self._load_profiles()
            return self.profiles
            
            
PROFILES = ProfilesIndex(os.environ["LAAB_PROFILES_ROOT"])