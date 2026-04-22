# Scripts

## `windows/`

PowerShell wrappers that let Windows 11 users invoke Contextd commands from PowerShell without typing `wsl` explicitly. Each is a one-line forward to the WSL-installed CLI.

**Customisation:** set `$env:CONTEXTD_WSL_DISTRO` (defaults to `Ubuntu`) or pass `-Distro <name>` on the command line to target a different WSL distro.

### Scripts

- `contextd-up.ps1` — start Memgraph + indexer
- `contextd-down.ps1` — stop
- `contextd-status.ps1` — report daemon + corpora
- `contextd-ask.ps1` — natural-language query
