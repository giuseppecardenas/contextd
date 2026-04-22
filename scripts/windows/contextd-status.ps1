param(
    [string]$Distro = $env:CONTEXTD_WSL_DISTRO
)
if (-not $Distro) { $Distro = "Ubuntu" }
wsl -d $Distro -- contextd status @Args
