# Streams per-process GPU usage on Windows as one JSON line every ~2 seconds.
# Uses the same performance counters as Task Manager's GPU columns.
$ErrorActionPreference = 'SilentlyContinue'
$paths = @('\GPU Engine(*)\Utilization Percentage', '\GPU Process Memory(*)\Dedicated Usage')
$info = @{}  # pid -> cached name / description / window title

while ($true) {
    $samples = (Get-Counter -Counter $paths -SampleInterval 1 -MaxSamples 1).CounterSamples
    $procs = @{}
    foreach ($s in $samples) {
        if ($s.InstanceName -notmatch '^pid_(\d+)_') { continue }
        $id = [int]$Matches[1]
        if (-not $procs.ContainsKey($id)) { $procs[$id] = @{ pid = $id; engines = @{}; vram = 0 } }
        if ($s.Path -like '*gpu process memory*') {
            $procs[$id].vram += $s.CookedValue
        } elseif ($s.InstanceName -match 'engtype_(.+)$') {
            # Task Manager sums each engine type, then shows the busiest one.
            $type = $Matches[1]
            $procs[$id].engines[$type] = $procs[$id].engines[$type] + $s.CookedValue
        }
    }
    $out = foreach ($p in $procs.Values) {
        $busy = ($p.engines.Values | Measure-Object -Maximum).Maximum
        if ($busy -lt 0.5 -and $p.vram -lt 200MB) { continue }
        if (-not $info.ContainsKey($p.pid)) {
            $gp = Get-Process -Id $p.pid
            if (-not $gp) { continue }
            $info[$p.pid] = @{
                name  = $gp.ProcessName
                desc  = $gp.MainModule.FileVersionInfo.FileDescription
                path  = $gp.Path
            }
        }
        $title = (Get-Process -Id $p.pid).MainWindowTitle
        [pscustomobject]@{
            pid = $p.pid; name = $info[$p.pid].name; desc = $info[$p.pid].desc
            path = $info[$p.pid].path; title = $title
            gpu = [math]::Round([double]$busy, 1)
            d3 = [math]::Round([double]$p.engines['3D'], 1)
            vram = [int64]$p.vram
        }
    }
    [Console]::Out.WriteLine((ConvertTo-Json -Compress -Depth 3 @{ t = [DateTimeOffset]::Now.ToUnixTimeSeconds(); apps = @($out) }))
    [Console]::Out.Flush()
    Start-Sleep -Milliseconds 800
}
