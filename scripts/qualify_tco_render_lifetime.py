"""Run the registered qualification with the inherited gradient-index convention."""

from contextlib import contextmanager
from unittest.mock import patch

import check_tco_render_lifetime as registered


class GradientIndexView:
    """Expose one-based update labels for the archived pre-optimizer hook."""

    def __init__(self, archive):
        self.archive = archive
        self.source = {}
        indices = set()
        for name in archive.files:
            label = name
            family, _, remainder = name.partition("/")
            if family == "grad":
                index, slash, factor = remainder.partition("/")
                if index not in {"0", "1"} or not slash or not factor:
                    raise ValueError(f"Unexpected clipped-gradient receipt key: {name}")
                indices.add(index)
                label = f"grad/{int(index) + 1}/{factor}"
            if label in self.source:
                raise ValueError(f"Duplicate normalized receipt key: {label}")
            self.source[label] = name
        if indices != {"0", "1"}:
            raise ValueError("Expected both zero-based clipped-gradient groups")
        self.files = list(self.source)

    def __getitem__(self, name):
        return self.archive[self.source[name]]


_registered_compare = registered.compare_npz
_registered_qualify = registered.qualify


def compare_npz(actual, expected, *, families=()):
    if "grad" not in families:
        return _registered_compare(actual, expected, families=families)

    import numpy as np

    load = np.load

    @contextmanager
    def load_indexed(path, **options):
        with load(path, **options) as archive:
            yield GradientIndexView(archive)

    # Use the registered inventory checks and numerical comparison on a read-only view.
    with patch.object(np, "load", load_indexed):
        return _registered_compare(actual, expected, families=families)


def qualify(include_fallback=False):
    with patch.object(registered, "compare_npz", compare_npz):
        result = _registered_qualify(include_fallback)
    result["qualification_reader_sha256"] = registered.sha(__file__, text=True)
    return result


def main():
    with patch.object(registered, "qualify", qualify):
        registered.main()


if __name__ == "__main__":
    main()
