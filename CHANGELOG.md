# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [2.0.1] - 2026-08-05
### Changed
- Normalize LLM-facing tool names to the OpenRouter-compatible character set, first-character rule, and 64-character limit while preserving original MCP names for calls.
- Always namespace colliding server/tool names; removed the unsafe collision-prefix opt-out.

## [2.0.0] - 2026-03-14
### Added
- Generic MCP Adapter with no dependencies on openai/gemini/etc.
### Changed
- Default install now doesn't install openai, thus the major version bump

## [1.0.0] - 2026-02-02
### Added
- Support for multiple MCPServers per adapter - with name collision handling and parallel invocation
- Github CI
### Changed
- Fully changed the API. There weren't any production users that we know of, so this breaking change is fine, no LTS for pre-1.0.0 version.
### Fixed

## [0.2.1] - 2026-02-01

### Added
### Changed

### Fixed
   - Fix trailing slash issues due to `httpx's` handling of `base_url`. Some MCP servers wouldn't work after a 307 trailing slash redirect due to headers strip or http method change. For example, Digital Ocean deployments are like that.


## [0.2.0] - 2026-02-01

### Added
   - MCP session initialization and caching caching for it
   - Multiple modes of initialization - on_fail, auto and none. Most of MCP servers I know and use don't fully follow the lifecycle, so auto might feel like an overkill for some.
### Changed

### Fixed


## [0.1.0] - 2026-01-31
### Added
   - Functional MCP requests without session initialization. Works with servers that don't fully follow lifecycle of MCP.
### Changed

### Fixed


## [0.0.5] - 2026-01-31
### Added
   - Gemini cli implementation.
### Changed

### Fixed
