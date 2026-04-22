param(
    [Parameter(Mandatory=$true)][string]$Question,
    [string]$Corpus = "",
    [string]$Distro = $env:CONTEXTD_WSL_DISTRO
)
if (-not $Distro) { $Distro = "Ubuntu" }
if ($Corpus) {
    wsl -d $Distro -- contextd ask "$Question" --corpus $Corpus
} else {
    wsl -d $Distro -- contextd ask "$Question"
}
