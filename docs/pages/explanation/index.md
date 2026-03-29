# Explanation

Background on design decisions, architecture, and the concepts behind Sklearn-Wrap.

| Section | Description |
|---------|-------------|
| [The Delegation Pattern](delegation-pattern.md) | Composition over inheritance, the three-phase lifecycle, and nested parameter syntax |
| [The Fit Context Lifecycle](fit-context-lifecycle.md) | How `_fit_context` automates instantiation, validation, and context management |
| [Trade-offs and Limitations](trade-offs.md) | Immutability, validation overhead, metadata routing, and proxy boundaries |
| [YAML Configuration Design](yaml-config-design.md) | Why declarative config, the trusted modules security model, and `!include` composition |
