# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).


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
