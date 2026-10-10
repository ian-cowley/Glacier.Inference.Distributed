<#
.SYNOPSIS
    Glacier High-Speed Model Downloader (PowerShell / .NET 10)
.DESCRIPTION
    Cross-platform PowerShell script running the pure C# multi-stream model downloader.
    Zero Python dependencies. Works on Windows, Linux, and macOS.
.EXAMPLE
    ./scripts/download_model.ps1 qwen2-57b
    ./scripts/download_model.ps1 mixtral-8x22b
    ./scripts/download_model.ps1 https://huggingface.co/path/to/model.gguf
#>
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$ModelOrUrl,

    [Parameter(Position = 1)]
    [string]$Dest = "models",

    [int]$Streams = 8
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$CliProj = Join-Path $RootDir "src\Glacier.Inference.Distributed.Cli\Glacier.Inference.Distributed.Cli.csproj"

& dotnet run --project "$CliProj" -c Release -- download $ModelOrUrl --dest $Dest --streams $Streams
