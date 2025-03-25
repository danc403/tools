@echo off
setlocal

rem Construct the command to run the Python script
set "python_script_path=%~dp0guardian2.py"
set "python_command=python "%python_script_path%" %1 %2"

rem Execute the Python script
%python_command%

endlocal
exit /b %errorlevel%
