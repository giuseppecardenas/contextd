# Scripts

## `windows/`

PowerShell wrappers for users running Contextd **inside WSL2**. Each script is a one-line forward to the WSL-installed CLI (`wsl -d <distro> -- contextd <cmd>`), so PowerShell users running the WSL2 alternative path don't have to type `wsl` explicitly.

**Native Windows users do not need these wrappers.** With a native install (`uv pip install -e .` or `pipx install contextd`), `contextd`, `contextd-mcp`, and `contextd-indexer` are on PATH as regular `.exe` shims under `.venv\Scripts\` — call them directly from PowerShell.

**Customisation (WSL2 setup only):** set `$env:CONTEXTD_WSL_DISTRO` (defaults to `Ubuntu`) or pass `-Distro <name>` on the command line to target a different WSL distro.

### Scripts

- `contextd-up.ps1` — start Memgraph + indexer
- `contextd-down.ps1` — stop
- `contextd-status.ps1` — report daemon + corpora
- `contextd-ask.ps1` — natural-language query
