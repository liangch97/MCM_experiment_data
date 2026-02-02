<#
.SYNOPSIS
    Full battery status continuous monitoring script (for SOC curve fitting)
.DESCRIPTION
    Collects: charge(mAh), temp, voltage, screen, brightness, network, GPS, top app
.EXAMPLE
    .\battery_full_monitor.ps1 -SceneName "scene2_video" -Duration 30
#>

param(
    [string]$DeviceIP = "192.168.1.51",
    [int]$Interval = 30,
    [int]$Duration = 30,
    [string]$SceneName = "test",
    [string]$OutputDir = ""
)

# Use script location as base if OutputDir not specified
if ([string]::IsNullOrEmpty($OutputDir)) {
    $OutputDir = Join-Path $PSScriptRoot "data"
}

$adb = Join-Path $PSScriptRoot "platform-tools\adb.exe"
$device = "${DeviceIP}:5555"
$totalSamples = [math]::Ceiling($Duration * 60 / $Interval)

# Create scene directory
$sceneDir = Join-Path $OutputDir $SceneName
if (-not (Test-Path $sceneDir)) {
    New-Item -ItemType Directory -Path $sceneDir -Force | Out-Null
}

$logFile = Join-Path $sceneDir "battery_monitor_log.csv"
$rawLogFile = Join-Path $sceneDir "raw_dumps.log"

# CSV header (cpu_util_pct = real CPU utilization 0-100%, gpu_util_pct from kgsl gpubusy)
"timestamp,elapsed_sec,charge_mAh,level_pct,voltage_mV,temp_C,screen,brightness,network_type,wifi_state,mobile_state,gps,top_app,cpu_util_pct,gpu_util_pct,gpu_mem_mb,gpu_frame_ms" | Out-File $logFile -Encoding UTF8

# Initialize CPU stat for utilization calculation
$prevCpuStat = $null

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Battery Full Monitor Started" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Device: $device"
Write-Host "Scene: $SceneName"
Write-Host "Interval: $Interval sec | Duration: $Duration min | Samples: ~$totalSamples"
Write-Host "Output: $logFile"
Write-Host "----------------------------------------------------------------"
Write-Host "Press Ctrl+C to stop"
Write-Host "================================================================"

$startTime = Get-Date
$sampleCount = 0

# Connect device
Write-Host "Connecting..." -ForegroundColor Yellow
$null = & $adb connect $device 2>&1
Start-Sleep -Seconds 1

try {
    while ($true) {
        $sampleCount++
        $now = Get-Date
        $elapsed = ($now - $startTime).TotalSeconds
        $elapsedMin = [math]::Round($elapsed / 60, 2)
        
        if ($elapsedMin -ge $Duration) {
            Write-Host "`n[DONE] Completed $Duration min" -ForegroundColor Green
            break
        }
        
        $timestamp = $now.ToString("yyyy-MM-dd HH:mm:ss")
        
        # 1. Battery data
        $batteryOutput = & $adb -s $device shell dumpsys battery 2>&1
        $batteryStr = $batteryOutput -join "`n"
        
        if ($batteryStr -match "error|unable|offline|cannot") {
            Write-Host "[WARN] Reconnecting..." -ForegroundColor Yellow
            $null = & $adb connect $device 2>&1
            Start-Sleep -Seconds 2
            continue
        }
        
        $level = ""; $charge_mAh = ""; $voltage = ""; $temp_C = ""
        if ($batteryStr -match "level:\s*(\d+)") { $level = $matches[1] }
        if ($batteryStr -match "Charge counter:\s*(\d+)") { $charge_mAh = [math]::Round([int]$matches[1] / 1000, 1) }
        # 匹配独立的 "voltage:" 行，排除 "Max charging voltage:"
        if ($batteryStr -match "(?m)^\s*voltage:\s*(\d+)") { $voltage = $matches[1] }
        if ($batteryStr -match "temperature:\s*(\d+)") { $temp_C = [math]::Round([int]$matches[1] / 10, 1) }
        
        # 2. Screen & Brightness (multiple methods for reliability)
        $displayOutput = & $adb -s $device shell dumpsys display 2>&1
        $displayStr = $displayOutput -join "`n"
        $screen = if ($displayStr -match "mScreenState=ON") { "on" } elseif ($displayStr -match "mScreenState=OFF") { "off" } else { "unknown" }
        
        # Brightness detection - try multiple methods
        $brightness = ""
        # Method 1: system setting (most reliable, 0-255 range)
        $brightnessOutput = & $adb -s $device shell settings get system screen_brightness 2>&1
        $brightnessStr = $brightnessOutput -join ""
        if ($brightnessStr -match "^(\d+)$") {
            $brightness = $matches[1]
        }
        # Method 2: fallback to dumpsys display mBrightness
        if ([string]::IsNullOrEmpty($brightness) -and $displayStr -match "mBrightness=(\d+)") {
            $brightness = $matches[1]
        }
        # Method 3: fallback to settings secure screen_brightness_float (0.0-1.0 -> convert to 0-255)
        if ([string]::IsNullOrEmpty($brightness)) {
            $brightnessFloatOutput = & $adb -s $device shell settings get secure screen_brightness_float 2>&1
            $brightnessFloatStr = $brightnessFloatOutput -join ""
            if ($brightnessFloatStr -match "^([\d.]+)$") {
                $brightness = [math]::Round([float]$matches[1] * 255, 0)
            }
        }
        
        # 3. Network
        $connOutput = & $adb -s $device shell dumpsys connectivity 2>&1
        $connStr = $connOutput -join "`n"
        $network_type = "none"
        if ($connStr -match "type:\s*WIFI") { $network_type = "wifi" }
        elseif ($connStr -match "NR_NSA|NR_SA|5G") { $network_type = "5g" }
        elseif ($connStr -match "LTE") { $network_type = "lte" }
        elseif ($connStr -match "MOBILE") { $network_type = "mobile" }
        
        $wifiOutput = & $adb -s $device shell dumpsys wifi 2>&1
        $wifiStr = $wifiOutput -join "`n"
        $wifi_state = if ($wifiStr -match "Wi-Fi is enabled") { "on" } else { "off" }
        
        $telOutput = & $adb -s $device shell dumpsys telephony.registry 2>&1
        $telStr = $telOutput -join "`n"
        $mobile_state = if ($telStr -match "mDataConnectionState=2") { "connected" } else { "off" }
        
        # 4. GPS (location_mode: 0=off, 1=sensors only, 2=battery saving, 3=high accuracy)
        $locOutput = & $adb -s $device shell settings get secure location_mode 2>&1
        $locStr = $locOutput -join ""
        $gps = if ($locStr -match "^[1-3]$") { "on" } else { "off" }
        
        # 5. Top app
        $actOutput = & $adb -s $device shell "dumpsys activity activities | grep mResumedActivity" 2>&1
        $actStr = $actOutput -join ""
        $top_app = ""
        if ($actStr -match "([a-zA-Z][a-zA-Z0-9_.]+)/") { $top_app = $matches[1] }
        
        # 6. CPU utilization (from /proc/stat, real 0-100%)
        # Format: cpu user nice system idle iowait irq softirq steal guest guest_nice
        $cpuOutput = & $adb -s $device shell cat /proc/stat 2>&1
        $cpuStr = $cpuOutput -join "`n"
        $cpu_util_pct = ""
        
        if ($cpuStr -match "cpu\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)") {
            $user = [long]$matches[1]
            $nice = [long]$matches[2]
            $system = [long]$matches[3]
            $idle = [long]$matches[4]
            $iowait = [long]$matches[5]
            $irq = [long]$matches[6]
            $softirq = [long]$matches[7]
            
            $currTotal = $user + $nice + $system + $idle + $iowait + $irq + $softirq
            $currIdle = $idle + $iowait
            
            if ($null -ne $prevCpuStat) {
                $deltaTotal = $currTotal - $prevCpuStat.Total
                $deltaIdle = $currIdle - $prevCpuStat.Idle
                if ($deltaTotal -gt 0) {
                    $cpu_util_pct = [math]::Round(100.0 * ($deltaTotal - $deltaIdle) / $deltaTotal, 1)
                }
            }
            
            # Store for next iteration
            $prevCpuStat = @{ Total = $currTotal; Idle = $currIdle }
        }
        
        # 7. GPU utilization (from /sys/class/kgsl/kgsl-3d0/gpubusy, Qualcomm KGSL)
        # Format: busy_cycles total_cycles (sliding window, resets after read)
        # Direct calculation: gpu_util = busy / total * 100%
        $gpu_util_pct = ""
        $gpuBusyOutput = & $adb -s $device shell cat /sys/class/kgsl/kgsl-3d0/gpubusy 2>&1
        $gpuBusyStr = $gpuBusyOutput -join ""
        if ($gpuBusyStr -match "^\s*(\d+)\s+(\d+)") {
            $busy = [long]$matches[1]
            $total = [long]$matches[2]
            
            if ($total -gt 0) {
                $gpu_util_pct = [math]::Round(100.0 * $busy / $total, 1)
                # Clamp to 0-100
                if ($gpu_util_pct -lt 0) { $gpu_util_pct = 0 }
                if ($gpu_util_pct -gt 100) { $gpu_util_pct = 100 }
            } else {
                # total=0 means GPU is idle
                $gpu_util_pct = 0
            }
        }
        
        # 8. GPU memory usage (from dumpsys gpu, Global total)
        $gpu_mem_mb = ""
        $gpuMemOutput = & $adb -s $device shell "dumpsys gpu 2>/dev/null | grep 'Global total'" 2>&1
        $gpuMemStr = $gpuMemOutput -join ""
        if ($gpuMemStr -match "Global total:\s*(\d+)") {
            $gpu_mem_mb = [math]::Round([long]$matches[1] / 1048576, 1)  # bytes to MB
        }
        
        # 9. GPU frame time (from gfxinfo of top app, 90th gpu percentile)
        $gpu_frame_ms = ""
        if (-not [string]::IsNullOrEmpty($top_app)) {
            $gfxOutput = & $adb -s $device shell "dumpsys gfxinfo $top_app 2>/dev/null | grep '90th gpu percentile'" 2>&1
            $gfxStr = $gfxOutput -join ""
            if ($gfxStr -match "90th gpu percentile:\s*(\d+)ms") {
                $gpu_frame_ms = $matches[1]
            }
        }
        
        # Write CSV row
        "$timestamp,$([int]$elapsed),$charge_mAh,$level,$voltage,$temp_C,$screen,$brightness,$network_type,$wifi_state,$mobile_state,$gps,$top_app,$cpu_util_pct,$gpu_util_pct,$gpu_mem_mb,$gpu_frame_ms" | Out-File $logFile -Append -Encoding UTF8
        
        # Write raw log
        "=== $timestamp ===" | Out-File $rawLogFile -Append -Encoding UTF8
        $batteryStr | Out-File $rawLogFile -Append -Encoding UTF8
        
        # Display
        $remaining = [math]::Round($Duration - $elapsedMin, 1)
        $color = if ([int]$level -lt 20) { "Red" } elseif ([int]$level -lt 50) { "Yellow" } else { "Green" }
        
        Write-Host "[$sampleCount/$totalSamples] $timestamp | " -NoNewline
        Write-Host "q=${charge_mAh}mAh SOC=${level}% T=${temp_C}C V=${voltage}mV " -ForegroundColor $color -NoNewline
        Write-Host "| cpu=${cpu_util_pct}% gpu=${gpu_util_pct}% scr=$screen net=$network_type | ${remaining}min" -ForegroundColor DarkGray
        
        if ($elapsedMin -lt $Duration) {
            Start-Sleep -Seconds $Interval
        }
    }
}
catch {
    Write-Host "`n[WARN] Interrupted: $_" -ForegroundColor Yellow
}
finally {
    $finalMin = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)
    
    Write-Host "`n================================================================" -ForegroundColor Cyan
    Write-Host "Summary: $sampleCount samples, $finalMin min" -ForegroundColor Green
    Write-Host "Data: $logFile"
    Write-Host "================================================================" -ForegroundColor Cyan
    
    # Export final stats
    Write-Host "Exporting final stats..." -ForegroundColor Yellow
    & $adb -s $device shell dumpsys batterystats > (Join-Path $sceneDir "batterystats_end.txt") 2>&1
    & $adb -s $device shell dumpsys battery > (Join-Path $sceneDir "battery_end.txt") 2>&1
    Write-Host "[OK] Export complete" -ForegroundColor Green
}
