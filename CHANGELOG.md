# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.0-alpha.7] - 2026-07-30

This **minor release** includes 18 commits.


### Features
- Prune defaults and use public class paths in captures  ([#75](https://github.com/stateful-y/sklearn-wrap/pull/75)) by @gtauzin

### Bug Fixes
- Pin exact uv version in setup-uv steps (template v0.29.6)  ([#63](https://github.com/stateful-y/sklearn-wrap/pull/63)) by @gtauzin
- Pin ossf/scorecard-action to v2.4.4 (@v2 tag does not exist) by @gtauzin
- Authenticate git-cliff's GitHub API calls  ([#76](https://github.com/stateful-y/sklearn-wrap/pull/76)) by @gtauzin

### Documentation
- Migrate docs engine from MkDocs to Zensical (template v0.30.1)  ([#64](https://github.com/stateful-y/sklearn-wrap/pull/64)) by @gtauzin

### Miscellaneous Tasks
- Update from template v0.26.1  ([#51](https://github.com/stateful-y/sklearn-wrap/pull/51)) by @gtauzin
- Update from template v0.27.0  ([#53](https://github.com/stateful-y/sklearn-wrap/pull/53)) by @gtauzin
- Update from template v0.27.3  ([#55](https://github.com/stateful-y/sklearn-wrap/pull/55)) by @gtauzin
- Update from template v0.28.1  ([#56](https://github.com/stateful-y/sklearn-wrap/pull/56)) by @gtauzin
- Update from template v0.28.3  ([#57](https://github.com/stateful-y/sklearn-wrap/pull/57)) by @gtauzin
- Update from template v0.28.4  ([#58](https://github.com/stateful-y/sklearn-wrap/pull/58)) by @gtauzin
- Update from template v0.29.3  ([#59](https://github.com/stateful-y/sklearn-wrap/pull/59)) by @gtauzin
- Update from template v0.31.1 (Renovate replaces Dependabot)  ([#66](https://github.com/stateful-y/sklearn-wrap/pull/66)) by @gtauzin
- Update to v0.32.1 (pre-push gates + CI roll-up)  ([#68](https://github.com/stateful-y/sklearn-wrap/pull/68)) by @gtauzin
- Sync to v0.35.0  ([#69](https://github.com/stateful-y/sklearn-wrap/pull/69)) by @gtauzin
- Sync to v0.36.0 (Codecov OIDC + scorecard pin)  ([#70](https://github.com/stateful-y/sklearn-wrap/pull/70)) by @gtauzin
- Sync to v0.37.0 (gitsign tag-signing docs)  ([#71](https://github.com/stateful-y/sklearn-wrap/pull/71)) by @gtauzin
- Update from python-package-copier v0.38.0  ([#72](https://github.com/stateful-y/sklearn-wrap/pull/72)) by @gtauzin

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
