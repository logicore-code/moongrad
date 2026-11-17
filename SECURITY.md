# Security policy

## Supported versions

moongrad is under active development and has not reached 1.0. The
current release line is the only one that receives fixes.

| Version | Supported |
|---|---|
| 0.7.x | yes |
| < 0.7 | no |

## Reporting a vulnerability

Please **do not** open a public issue for a security problem. Use
GitHub's private vulnerability reporting on this repository
(the "Security" tab → "Report a vulnerability"); that opens a
private channel with the maintainer.

Please include:

- what the problem is, and which file and function it is in;
- how to reproduce it — a failing test is ideal;
- what an attacker or a caller could do with it.

You can expect an acknowledgement within a week. If the report is
accepted, a fix will be prepared and released, and you will be
credited in the changelog unless you would rather not be.

## Scope

This library computes derivatives. It reads no files outside the two
modules that do so deliberately (`model_io.mbt` for JSON,
`dataset.mbt` for data), opens no network connections, has no FFI,
and depends only on the MoonBit standard library. It does not run
untrusted input in any privileged position, so the realistic classes
of problem are memory-safety or arithmetic faults in the numerical
code — a shape mismatch that is not refused, a division that produces
a NaN where the caller believes it produced a number, or an
out-of-bounds access.
