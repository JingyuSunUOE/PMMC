"""Emit pip constraints for the running image's branch of the public uv.lock."""

import sys
import tomllib
from pathlib import Path

from packaging.markers import Marker


def constraints(path: Path) -> list[str]:
    packages = tomllib.loads(path.read_text())["package"]
    pins = {}
    for package in packages:
        if "registry" not in package["source"]:
            if package["name"] != "pmmc":
                raise ValueError("non-registry release dependency: " + package["name"])
            continue
        markers = package.get("resolution-markers", [])
        if markers and not any(Marker(marker).evaluate() for marker in markers):
            continue
        name, version = package["name"], package["version"]
        if name in pins and pins[name] != version:
            raise ValueError("ambiguous release dependency: " + name)
        pins[name] = version
    if not pins:
        raise ValueError("empty release lock")
    return [f"{key}=={pins[key]}" for key in sorted(pins)]


if __name__ == "__main__":
    print("\n".join(constraints(Path(sys.argv[1]))))
