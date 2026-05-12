# Contributing

Thanks for your interest in `bachelor-thesis-flow`. This document describes
the workflow expected for changes.

## Getting started

```bash
git clone https://github.com/rayanchatt/bachelor-thesis-flow.git
cd bachelor-thesis-flow
make install   # editable install + dev tools + pre-commit hooks
```

The supported Python versions are **3.11** and **3.12**.

## Workflow

1. Create a feature branch off `main`.
2. Make your changes.
3. Keep the package healthy locally:

   ```bash
   make lint       # ruff (lint)
   make format     # ruff (format) — applies fixes
   make typecheck  # mypy --strict
   make test       # pytest
   make cov        # pytest with coverage
   ```

   `pre-commit` runs the same checks at commit time; CI runs them on every
   push and pull request across Python 3.11/3.12 on ubuntu and macos.

4. Open a pull request against `main`.

## Commit messages

Commit messages **must** follow
[Conventional Commits](https://www.conventionalcommits.org/). The release
automation parses them to generate `CHANGELOG.md` and the next version tag.

| Prefix      | Use for                                                            |
|-------------|--------------------------------------------------------------------|
| `feat:`     | A new user-facing feature.                                          |
| `fix:`      | A user-facing bug fix.                                              |
| `perf:`     | A performance improvement with no behaviour change.                 |
| `refactor:` | Internal refactor with no behaviour change.                         |
| `docs:`     | Documentation only.                                                 |
| `test:`     | Tests only.                                                         |
| `build:`    | Build system or package metadata.                                   |
| `ci:`       | CI / GitHub Actions / pre-commit.                                   |
| `chore:`    | Anything else that does not affect runtime behaviour.               |

Breaking changes use either a `!` after the type (e.g. `refactor!:`) or a
`BREAKING CHANGE:` footer in the commit body.

## Code style

- **Formatter**: `ruff format` (line length 100, double quotes).
- **Linter**: `ruff check` with the rule sets declared in `pyproject.toml`.
- **Type checker**: `mypy --strict` for everything under `src/btflow/`.
- **Imports** are sorted by ruff's `I` rule set.
- **Docstrings**: Google-style. Public APIs require docstrings; private
  helpers do not.
- Avoid bare `except`; prefer narrow exception types.

## Tests

- New code requires tests. Aim to keep coverage at or above the current
  baseline (~84 %).
- Use the synthetic fixtures in `tests/conftest.py` rather than depending
  on real microscopy data so CI stays self-contained.

## Reporting issues

Please use the [bug report](.github/ISSUE_TEMPLATE/bug_report.yml) or
[feature request](.github/ISSUE_TEMPLATE/feature_request.yml) issue
templates.

Security-sensitive reports should go through GitHub's
[Security Advisories](https://github.com/rayanchatt/bachelor-thesis-flow/security/advisories/new)
flow, not the public issue tracker. See [`SECURITY.md`](SECURITY.md).
