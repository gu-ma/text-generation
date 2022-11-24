@echo off

@REM Set a variable with additional command line arguments
:loop
set args=%args% %1
shift
if not "%~1"=="" goto loop

@REM Run docker command
SET mypath=%~dp0
docker run %args% -p 5004:5004 -it -v %mypath%:/scratch transformer-text-gen:latest bash -c "(cd /scratch/ && python api.py --path=models/salvatore --no_cuda)"