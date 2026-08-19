#!/usr/bin/env pwsh
# Start the analysis server and the web dev server together.
#
#   ./dev.ps1                 both, waits for the server before vite starts
#   ./dev.ps1 -NoServer       web only (same as: cd web; npm run dev)
#   ./dev.ps1 -- -v           everything after -- goes to terrarium-server
#
# Ctrl-C in this console stops both. The server needs the `server` extra
# installed in analysis/.venv — see README.md.

[CmdletBinding()]
param(
  [switch]$NoServer,
  [int]$Port = 8765,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ServerArgs = @()
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$server = $null

function Test-ServerPort {
  try {
    $c = [System.Net.Sockets.TcpClient]::new()
    $c.Connect('127.0.0.1', $Port)
    $c.Close()
    return $true
  } catch { return $false }
}

try {
  if (-not $NoServer) {
    if (Test-ServerPort) {
      Write-Host "analysis server already listening on $Port — leaving it alone" -ForegroundColor DarkGray
    } else {
      if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is not on PATH. Install it, or run with -NoServer for the web app alone."
      }
      $argList = @('run', '--extra', 'server', 'terrarium-server', '--port', $Port) + $ServerArgs
      $server = Start-Process -PassThru -NoNewWindow -WorkingDirectory "$root/analysis" `
        -FilePath 'uv' -ArgumentList $argList

      # The browser probes the server exactly once at startup, and cold-loading
      # the models takes a while — so hold vite back until the port answers.
      Write-Host "waiting for the analysis server on $Port ..." -ForegroundColor DarkGray
      $deadline = (Get-Date).AddSeconds(180)
      while (-not (Test-ServerPort)) {
        if ($server.HasExited) { throw "terrarium-server exited with code $($server.ExitCode)" }
        if ((Get-Date) -gt $deadline) {
          Write-Host "server still not up after 180s — starting vite anyway" -ForegroundColor Yellow
          break
        }
        Start-Sleep -Milliseconds 500
      }
    }
  }

  Push-Location "$root/web"
  npm run dev
} finally {
  Pop-Location -ErrorAction SilentlyContinue
  if ($server -and -not $server.HasExited) {
    # uv spawns python as a child; kill the tree, not just the launcher.
    if (Get-Command taskkill -ErrorAction SilentlyContinue) {
      taskkill /T /F /PID $server.Id | Out-Null
    } else {
      Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    }
  }
}
