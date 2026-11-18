"""Logical artifact filenames shared by Linux containers and the Windows host.

Docker Desktop exposes a Linux filename's colon as U+F03A on this Windows bind
mount. Only that explicit Windows mapping is applied. All other Unicode code
points, case, spacing and file bytes remain unchanged; Linux uses identity.

``logical_inventory(root)`` returns {logical_POSIX_relative_name: physical_Path}
for existing files. ``resolve_logical(root, name)`` returns a physical Path and
also supports a new output path; pass ``must_exist=True`` for an existing file.
Both reject traversal, drive paths, and ambiguous codec collisions. Neither
function creates, renames, rewrites, hashes, or deletes anything.
"""
import os
from pathlib import Path
import re


WINDOWS_COLON = "\uf03a"


def _windows(value):
    if value is not None and not isinstance(value, bool):
        raise TypeError("windows must be a bool or None")
    return os.name == "nt" if value is None else value


def _parts(name, windows):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"Expected a nonempty canonical relative POSIX artifact name: {name!r}")
    parts = name.split("/")
    if any(part in ("", ".", "..") or re.match(r"^[A-Za-z]:", part) for part in parts):
        raise ValueError(f"Absolute, drive, empty or traversal component in artifact name: {name!r}")
    if windows and WINDOWS_COLON in name:
        raise ValueError("Windows logical names must use ':'; caller-supplied U+F03A is ambiguous")
    return parts


def _logical_component(name, windows):
    logical = name.replace(WINDOWS_COLON, ":") if windows else name
    if len(_parts(logical, windows)) != 1:
        raise ValueError(f"Physical directory entry is not one safe component: {name!r}")
    return logical


def _index_components(entries, windows):
    """Index real sibling entries; kept pure so collisions can be tested on any OS."""
    result = {}
    for entry in entries:
        name = _logical_component(entry.name, windows)
        if name in result:
            raise ValueError(f"Logical artifact filename collision: {result[name]} and {entry} -> {name!r}")
        result[name] = entry
    return result


def _contained(root, path):
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Artifact path escapes its root, including through a link: {path}")
    return resolved


def logical_inventory(root, *, windows=None):
    """Return deterministic logical POSIX names mapped to existing physical files.

    Decode the Windows colon mapping before inventory comparison. Reject even
    directory-level normalization collisions, external links, and cyclic or
    duplicate directory aliases. Empty directories are not evidence files.
    """
    windows = _windows(windows)
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Artifact inventory root is not an existing directory: {root}")
    result, pending, visited = {}, [(root, "")], set()
    while pending:
        directory, prefix = pending.pop()
        actual = _contained(root, directory)
        if actual in visited:
            raise ValueError(f"Duplicate or cyclic artifact directory alias: {directory}")
        visited.add(actual)
        entries = _index_components(directory.iterdir(), windows)
        for component in sorted(entries):
            path = entries[component]
            _contained(root, path)
            logical = prefix + component
            if path.is_dir():
                pending.append((path, logical + "/"))
            elif path.is_file():
                if logical in result:
                    raise ValueError(f"Logical artifact inventory collision: {logical!r}")
                result[logical] = path
            else:
                raise ValueError(f"Unsupported or broken artifact entry: {path}")
    return dict(sorted(result.items()))


def resolve_logical(root, name, *, windows=None, must_exist=False):
    """Resolve a canonical POSIX relative name without allowing root escape.

    Existing ancestors are inspected so aliases/collisions cannot silently pick
    a different file. For missing destinations, encode ':' as U+F03A on Windows.
    Linux names (including literal U+F03A) remain identical to their inputs.
    """
    windows = _windows(windows)
    parts = _parts(name, windows)
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Artifact resolution root is not an existing directory: {root}")
    path = root
    for index, component in enumerate(parts):
        if path.exists():
            if not path.is_dir():
                raise ValueError(f"Artifact ancestor is not a directory: {path}")
            entries = _index_components(path.iterdir(), windows)
            physical = component.replace(":", WINDOWS_COLON) if windows else component
            path = entries.get(component, path / physical)
        else:
            path = path / (component.replace(":", WINDOWS_COLON) if windows else component)
        _contained(root, path)
        if index < len(parts)-1 and path.exists() and not path.is_dir():
            raise ValueError(f"Artifact ancestor is not a directory: {path}")
    if must_exist and not path.is_file():
        raise ValueError(f"Missing existing artifact file: {name}")
    return path
