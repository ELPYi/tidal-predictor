"""
Station profile data model and JSON persistence.

Profiles are saved as JSON files in the 'profiles/' directory
next to this module.
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# When frozen by PyInstaller, save profiles next to the exe (not in temp bundle)
if getattr(sys, 'frozen', False):
    _APP_DIR = os.path.dirname(sys.executable)
else:
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))

PROFILES_DIR = os.path.join(_APP_DIR, "profiles")


@dataclass
class Constituent:
    name: str
    amplitude: float
    phase: float  # degrees


@dataclass
class Station:
    name: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    datum_name: str = "MLLW"
    datum_offset: float = 0.0
    timezone_label: str = "UTC"
    constituents: List[Constituent] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Station":
        constituents = [Constituent(**c) for c in d.get("constituents", [])]
        return cls(
            name=d.get("name", ""),
            latitude=d.get("latitude", 0.0),
            longitude=d.get("longitude", 0.0),
            datum_name=d.get("datum_name", "MLLW"),
            datum_offset=d.get("datum_offset", 0.0),
            timezone_label=d.get("timezone_label", "UTC"),
            constituents=constituents,
        )

    def save(self, filename: Optional[str] = None) -> str:
        """Save station profile to JSON. Returns the file path."""
        os.makedirs(PROFILES_DIR, exist_ok=True)
        if filename is None:
            # Sanitize station name for filename
            safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in self.name)
            safe = safe.strip() or "station"
            filename = safe + ".json"
        filepath = os.path.join(PROFILES_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath

    @classmethod
    def load(cls, filename: str) -> "Station":
        """Load station profile from JSON file in profiles directory."""
        filepath = os.path.join(PROFILES_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def list_profiles() -> List[str]:
    """Return list of saved profile filenames."""
    os.makedirs(PROFILES_DIR, exist_ok=True)
    return sorted(f for f in os.listdir(PROFILES_DIR) if f.endswith(".json"))


def delete_profile(filename: str) -> None:
    """Delete a saved profile."""
    filepath = os.path.join(PROFILES_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
