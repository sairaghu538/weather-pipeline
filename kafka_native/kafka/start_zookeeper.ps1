$ErrorActionPreference = "Stop"
$kafkaDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Use wildcard classpath (supported by Java 6+) to avoid long command lines
$classpath = "$kafkaDir\libs\*;$kafkaDir\config"
$log4jOpts = "-Dlog4j.configuration=file:$kafkaDir\config\log4j.properties"
$mainClass = "org.apache.zookeeper.server.quorum.QuorumPeerMain"
$configFile = "$kafkaDir\config\zookeeper.properties"

Write-Host "Starting Zookeeper..."
Write-Host "Classpath: $classpath"

# Pass heap options as separate arguments
& java -Xmx512M -Xms512M $log4jOpts -cp $classpath $mainClass $configFile
