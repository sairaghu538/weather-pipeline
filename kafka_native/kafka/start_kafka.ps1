$ErrorActionPreference = "Stop"
$kafkaDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Use wildcard classpath (supported by Java 6+) to avoid long command lines
$classpath = "$kafkaDir\libs\*;$kafkaDir\config"
$log4jOpts = "-Dlog4j.configuration=file:$kafkaDir\config\log4j.properties"
$mainClass = "kafka.Kafka"
$configFile = "$kafkaDir\config\server.properties"

Write-Host "Starting Kafka Broker..."
Write-Host "Classpath: $classpath"

# Pass heap options as separate arguments
& java -Xmx1G -Xms1G $log4jOpts -cp $classpath $mainClass $configFile
