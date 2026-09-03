# Security Policy

## Supported versions

Security and data-integrity fixes are applied to the latest tagged or documented
release on `master` (currently 0.14.1). Older versions are not maintained as
separate security branches. Install from this repository; the PyPI name
`momentum-lab` belongs to an unrelated project.

Keep older source/version-locked research artifacts with their original
environment. An upgrade requires a new run or protocol, not bypassing resume
checks or deleting observation history. Never load untrusted checkpoints.

## Reporting a vulnerability

Please use GitHub's private security-advisory workflow when available. Do not
open a public issue for credential exposure, arbitrary code execution, path
traversal, dependency compromise, or a data-integrity flaw that could silently
change research results.

Include affected versions, reproduction steps, expected impact, and any safe
mitigation. Never attach real brokerage credentials, API secrets, or private
market data.

## Scope

Relevant reports include software vulnerabilities, unsafe package/distribution
behavior, checkpoint tampering, path handling, dependency supply-chain risks,
and silent research-data corruption. Disagreement with a trading strategy's
future performance is not a security issue.
