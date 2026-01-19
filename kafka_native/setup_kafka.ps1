$ErrorActionPreference = "Stop"
$kafkaVer = "3.6.1"
$scalaVer = "2.13"
$zipName = "kafka_$scalaVer-$kafkaVer.tgz"
$url = "https://archive.apache.org/dist/kafka/$kafkaVer/$zipName"
$destDir = "$PSScriptRoot"

Write-Host "1. Downloading Kafka $kafkaVer from $url..."
# Use TLS 1.2
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri $url -OutFile "$destDir\$zipName"

Write-Host "2. Extracting archive..."
# Native tar in Windows 10/11
tar -xf "$destDir\$zipName" -C "$destDir"

$extractedDir = "$destDir\kafka_$scalaVer-$kafkaVer"
$finalDir = "$destDir\kafka"

if (Test-Path $finalDir) { 
    Write-Host "Removing existing kafka folder..."
    Remove-Item -Recurse -Force $finalDir 
}
Rename-Item -Path $extractedDir -NewName "kafka"

Write-Host "3. Configuring paths for Windows..."
$serverProp = "$finalDir\config\server.properties"
$zooProp = "$finalDir\config\zookeeper.properties"

# Fix Kafka Logs path (use forward slashes for Windows Java compatibility)
$txt = Get-Content $serverProp
$txt = $txt -replace "log.dirs=/tmp/kafka-logs", "log.dirs=./kafka-logs"
Set-Content $serverProp $txt

# Fix Zookeeper Data path
$txt = Get-Content $zooProp
$txt = $txt -replace "dataDir=/tmp/zookeeper", "dataDir=./zookeeper-data"
Set-Content $zooProp $txt

Write-Host "4. Creating launcher scripts..."
$batContentZoo = "bin\windows\zookeeper-server-start.bat config\zookeeper.properties"
Set-Content "$finalDir\run_zookeeper.bat" $batContentZoo

$batContentKafka = "bin\windows\kafka-server-start.bat config\server.properties"
Set-Content "$finalDir\run_kafka.bat" $batContentKafka

# Cleanup
Remove-Item "$destDir\$zipName"

Write-Host "---------------------------------------------------"
Write-Host "SETUP COMPLETE!"
Write-Host "To start Kafka, you need TWO terminal windows:"
Write-Host "1) cd kafka_native\kafka ; .\run_zookeeper.bat"
Write-Host "2) cd kafka_native\kafka ; .\run_kafka.bat"
Write-Host "---------------------------------------------------"
