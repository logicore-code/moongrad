#!/usr/bin/env python3
"""Check that the toolchain is new enough for this project.

The monthly competition requires the compiler (`moonc`) to be at
least 1.0.0-rc1. The toolchain cannot be pinned to a release
download -- the versioned archives are not served -- so the check
has to be made at build time instead: refuse to build with a
compiler that is too old, and print the version that was found so
the build log says which one it was.

The check is a script rather than a shell one-liner so that it can
be run locally, in CI, and by a reviewer, and give the same
answer in all three places.

    python3 scripts/check_toolchain.py [--min 1.0.0-rc1]

Exit status is 0 when the compiler is new enough, 1 otherwise.
"""

import argparse
import re
import subprocess
import sys

# The compiler version the project is accepted with, and the one the
# source needs at minimum. They are the same for now.
DEFAULT_MIN = "1.0.0-rc1"

VERSION_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.]+))?")


def read_moonc_version():
    """Return the version string of `moonc`, or None if it is unknown."""
    for cmd in (["moonc", "-v"], ["moon", "version", "--all"]):
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for line in out.splitlines():
            if "moonc" not in line and cmd[0] != "moonc":
                continue
            m = VERSION_RE.search(line)
            if m:
                return "%s.%s.%s%s" % (
                    m.group(1),
                    m.group(2),
                    m.group(3),
                    "-" + m.group(4) if m.group(4) else "",
                )
    return None


def parse(version):
    """Split a version into (numbers, pre-release tags)."""
    m = VERSION_RE.search(version)
    if not m:
        return None, None
    nums = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    pre = m.group(4)
    tags = pre.split(".") if pre else []
    return nums, tags


def pre_rank(tags):
    """Rank a pre-release tag list against the `rc1` marker.

    An empty list is a final release, which is newer than any
    pre-release. `rc1` is the marker itself. `alpha` and `beta` are
    older than any `rc`.
    """
    if not tags:
        return 2  # final release
    first = tags[0]
    if first == "rc":
        try:
            return (1, int(tags[1]) if len(tags) > 1 else 0)
        except ValueError:
            return (1, 0)
    if first in ("alpha", "beta"):
        try:
            return (0, int(tags[1]) if len(tags) > 1 else 0)
        except ValueError:
            return (0, 0)
    return (1, 0)


def at_least(version, minimum):
    """True when `version` is at least `minimum`."""
    vnums, vtags = parse(version)
    mnums, mtags = parse(minimum)
    if vnums is None or mnums is None:
        return False
    if vnums != mnums:
        return vnums > mnums
    vr, mr = pre_rank(vtags), pre_rank(mtags)
    if vr[0] != mr[0]:
        return vr[0] > mr[0]
    return vr[1] >= mr[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", default=DEFAULT_MIN, dest="minimum")
    args = ap.parse_args()

    version = read_moonc_version()
    if version is None:
        print("::error::the moonc version could not be read")
        return 1

    print("moonc is %s" % version)
    if not at_least(version, args.minimum):
        print(
            "::error::moonc %s is older than %s; this project needs %s or newer"
            % (version, args.minimum, args.minimum)
        )
        return 1

    print("moonc %s satisfies the requirement of %s or newer" % (version, args.minimum))
    print("moonc_version=%s" % version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
