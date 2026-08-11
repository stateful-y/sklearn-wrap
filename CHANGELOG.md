# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.0-alpha.8] - 2026-08-10

This **minor release** includes 6 commits.


### Bug Fixes
- Stop the nightly coverage upload from silently uploading the wrong report  ([#90](https://github.com/stateful-y/sklearn-wrap/pull/90)) by @gtauzin

### Refactoring
- Move throwaway build output to .artifacts/ and config to .github/  ([#89](https://github.com/stateful-y/sklearn-wrap/pull/89)) by @gtauzin

### Miscellaneous Tasks
- Fix three release-pipeline defects (template v0.39.0)  ([#82](https://github.com/stateful-y/sklearn-wrap/pull/82)) by @gtauzin
- Let Renovate see the SBOM tool's version pin (template v0.39.1)  ([#83](https://github.com/stateful-y/sklearn-wrap/pull/83)) by @gtauzin
- Add a nightly job that exercises the release path (template v0.40.0)  ([#84](https://github.com/stateful-y/sklearn-wrap/pull/84)) by @gtauzin
- Fix a shell injection in the release publish job (template v0.40.1)  ([#88](https://github.com/stateful-y/sklearn-wrap/pull/88)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.7] - 2026-07-30

This **minor release** includes 18 commits.


### Features
- Capture only explicitly set parameters, and classes at their public import path  ([#75](https://github.com/stateful-y/sklearn-wrap/pull/75)) by @gtauzin

### Bug Fixes
- Install git hooks with prek install -f so an existing hook is replaced  ([#58](https://github.com/stateful-y/sklearn-wrap/pull/58)) by @gtauzin
- Pin an exact uv version in every CI workflow  ([#63](https://github.com/stateful-y/sklearn-wrap/pull/63)) by @gtauzin
- Pin ossf/scorecard-action to v2.4.4 (@v2 tag does not exist) by @gtauzin
- Stop the changelog workflow failing on GitHub API rate limits  ([#76](https://github.com/stateful-y/sklearn-wrap/pull/76)) by @gtauzin

### Documentation
- Build See Also links from each page's own depth  ([#51](https://github.com/stateful-y/sklearn-wrap/pull/51)) by @gtauzin
- Stop publishing the docs build tooling's source on the documentation site  ([#55](https://github.com/stateful-y/sklearn-wrap/pull/55)) by @gtauzin
- Warn instead of silently dropping API entries that cannot be resolved  ([#57](https://github.com/stateful-y/sklearn-wrap/pull/57)) by @gtauzin
- Fix a See Also cross-reference that broke the strict documentation build  ([#59](https://github.com/stateful-y/sklearn-wrap/pull/59)) by @gtauzin
- Improve See Also rendering for external packages and name-only targets  ([#59](https://github.com/stateful-y/sklearn-wrap/pull/59)) by @gtauzin
- Migrate the documentation engine from MkDocs to Zensical  ([#64](https://github.com/stateful-y/sklearn-wrap/pull/64)) by @gtauzin
- Add a page describing the project's security posture  ([#69](https://github.com/stateful-y/sklearn-wrap/pull/69)) by @gtauzin
- Document signing release tags with gitsign  ([#71](https://github.com/stateful-y/sklearn-wrap/pull/71)) by @gtauzin

### Refactoring
- Split the docs build tooling out of a single hooks module  ([#55](https://github.com/stateful-y/sklearn-wrap/pull/55)) by @gtauzin
- Build API pages from mkdocstrings templates instead of rewriting HTML  ([#56](https://github.com/stateful-y/sklearn-wrap/pull/56)) by @gtauzin
- Discover the public API with Griffe instead of hand-rolled AST analysis  ([#57](https://github.com/stateful-y/sklearn-wrap/pull/57)) by @gtauzin
- Move the docs build tooling from docs/ to docs_build/  ([#59](https://github.com/stateful-y/sklearn-wrap/pull/59)) by @gtauzin

### Miscellaneous Tasks
- Switch the git hook runner from pre-commit to prek  ([#53](https://github.com/stateful-y/sklearn-wrap/pull/53)) by @gtauzin
- Check that pull request commit messages follow Conventional Commits  ([#53](https://github.com/stateful-y/sklearn-wrap/pull/53)) by @gtauzin
- Replace Dependabot with Renovate for dependency updates  ([#66](https://github.com/stateful-y/sklearn-wrap/pull/66)) by @gtauzin
- Run the slower lint gates at pre-push instead of on every commit  ([#68](https://github.com/stateful-y/sklearn-wrap/pull/68)) by @gtauzin
- Add a single CI passed roll-up job for the required checks  ([#68](https://github.com/stateful-y/sklearn-wrap/pull/68)) by @gtauzin
- Seed a CLAUDE.md with project instructions for AI coding assistants  ([#72](https://github.com/stateful-y/sklearn-wrap/pull/72)) by @gtauzin

### Security
- Restrict GitHub Actions workflows to read-only permissions by default  ([#69](https://github.com/stateful-y/sklearn-wrap/pull/69)) by @gtauzin
- Scan for hardcoded secrets from a git hook and a CI job  ([#69](https://github.com/stateful-y/sklearn-wrap/pull/69)) by @gtauzin
- Add CodeQL and OpenSSF Scorecard analysis workflows  ([#69](https://github.com/stateful-y/sklearn-wrap/pull/69)) by @gtauzin
- Enable ruff's flake8-bandit rules on the source tree  ([#69](https://github.com/stateful-y/sklearn-wrap/pull/69)) by @gtauzin
- Attach a CycloneDX SBOM to published release artifacts  ([#69](https://github.com/stateful-y/sklearn-wrap/pull/69)) by @gtauzin
- Add a security policy and a CODEOWNERS file  ([#69](https://github.com/stateful-y/sklearn-wrap/pull/69)) by @gtauzin
- Upload coverage with GitHub OIDC instead of a stored Codecov token  ([#70](https://github.com/stateful-y/sklearn-wrap/pull/70)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.6] - 2026-04-19

This **minor release** includes 4 commits.


### Features
- Add EstimatorConfig for YAML-based estimator configuration  ([#38](https://github.com/stateful-y/sklearn-wrap/pull/38)) by @gtauzin

### Bug Fixes
- Avoid marimo name mangling of _fit_context  ([#43](https://github.com/stateful-y/sklearn-wrap/pull/43)) by @gtauzin

### Miscellaneous Tasks
- Update from template 0.16.0  ([#37](https://github.com/stateful-y/sklearn-wrap/pull/37)) by @gtauzin
- Update from template v0.18.0 and restructure docs with Diataxis  ([#40](https://github.com/stateful-y/sklearn-wrap/pull/40)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.5] - 2026-03-01

This **minor release** includes 2 commits.


### Features
- Update from template v0.14.0 with PEP 723 and marimo playground links  ([#30](https://github.com/stateful-y/sklearn-wrap/pull/30)) by @gtauzin
- Update from template v0.15.0 with API auto-gen, gallery system, and docs improvements  ([#33](https://github.com/stateful-y/sklearn-wrap/pull/33)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.4] - 2026-02-23

This **minor release** includes 8 commits.


### Documentation
- Update README.md  ([#19](https://github.com/stateful-y/sklearn-wrap/pull/19) and [#21](https://github.com/stateful-y/sklearn-wrap/pull/21)) by @gtauzin
- Export examples as both static HTML and interactive WASM  ([#24](https://github.com/stateful-y/sklearn-wrap/pull/24)) by @gtauzin
- Reformulate docs text  ([#27](https://github.com/stateful-y/sklearn-wrap/pull/27)) by @gtauzin

### Miscellaneous Tasks
- Update from copier template  ([#22](https://github.com/stateful-y/sklearn-wrap/pull/22), [#25](https://github.com/stateful-y/sklearn-wrap/pull/25), and [#28](https://github.com/stateful-y/sklearn-wrap/pull/28)) by @gtauzin
- Align notebook examples with contributing guidelines  ([#25](https://github.com/stateful-y/sklearn-wrap/pull/25)) by @gtauzin
- Guard codecov steps when token is unavailable  ([#26](https://github.com/stateful-y/sklearn-wrap/pull/26)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.3] - 2026-02-10

This **minor release** includes 1 commit.


### Bug Fixes
- Add pyodide package install cells to marimo notebooks  ([#17](https://github.com/stateful-y/sklearn-wrap/pull/17)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.2] - 2026-02-09

This **minor release** includes 4 commits.


### Documentation
- Update README with theme-aware logo and stateful-y branding  ([#12](https://github.com/stateful-y/sklearn-wrap/pull/12)) by @gtauzin

### Refactoring
- Enforce `_estimator_name` as constructor keyword  ([#13](https://github.com/stateful-y/sklearn-wrap/pull/13)) by @gtauzin

### Miscellaneous Tasks
- Update GitHub Actions to latest versions  ([#11](https://github.com/stateful-y/sklearn-wrap/pull/11)) by @gtauzin
- Replace template placeholders with actual project values  ([#14](https://github.com/stateful-y/sklearn-wrap/pull/14)) by @gtauzin

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [0.1.0-alpha.1] - 2026-02-07

This **minor release** includes 1 commit.

- Initial commit

### Contributors

Thanks to all contributors for this release:
- @gtauzin

## [Unreleased]

### Added
- Initial project setup
