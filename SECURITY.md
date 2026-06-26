# Security Policy

## Supported Versions

Security fixes are provided for the latest release and the `main` branch.

| Version | Supported |
|---------|-----------|
| latest  | ✅ |
| main    | ✅ |
| older   | ❌ |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please use [GitHub's private vulnerability reporting](https://github.com/kent-tokyo/causasv/security/advisories/new).

Include:
- A clear description of the issue
- Steps to reproduce
- Affected versions or commits
- Potential impact and suggested mitigation

**Response timeline:** I will try to acknowledge valid reports within 72 hours.
After triage, I will provide a fix, a workaround, or a request for more information.

## Disclosure Policy

Please allow reasonable time to investigate and fix before public disclosure.
A GitHub Security Advisory and patched release will be published when a fix is available.

## Scope

**In scope:**
- Unsafe Rust behavior that can lead to memory unsafety
- Undefined behavior
- Dependency vulnerabilities
- Python binding issues causing crashes or unsafe behavior
- CI/CD or release process vulnerabilities

**Out of scope:**
- General feature requests or non-security bugs
- Performance issues without security impact
- Vulnerabilities requiring malicious local code execution in an already-compromised environment
