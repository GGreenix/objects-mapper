@echo off
rem Double-click to start FRC Field Mapper. The first run sets everything up.
setlocal EnableExtensions
title FRC Field Mapper
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"
if exist "%VENV_PY%" goto run

echo Setting up Field Mapper for the first time...
set "PY="
call :trypy 3.11
call :trypy 3.12
call :trypy 3.13
if not defined PY call :trypath
if not defined PY goto nopython

echo Creating the Python environment with %PY% ...
%PY% -m venv .venv
if errorlevel 1 goto venvfail

:run
"%VENV_PY%" launch.py %*
if errorlevel 1 goto failed
exit /b 0

:trypy
if defined PY exit /b 0
py -%1 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY=py -%1"
exit /b 0

:trypath
python -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
exit /b 0

:nopython
echo.
echo Python 3.11 is needed but was not found.
where winget >nul 2>&1
if errorlevel 1 goto manualpython
choice /C YN /M "Install Python 3.11 now"
if errorlevel 2 goto manualpython
winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
echo.
echo Python is installed. Close this window and double-click start.bat again.
pause
exit /b 1

:manualpython
echo Install Python 3.11 from python.org (tick "Add python.exe to PATH"),
echo then double-click start.bat again.
start "" "https://www.python.org/downloads/windows/"
pause
exit /b 1

:venvfail
echo Could not create the Python environment in .venv
pause
exit /b 1

:failed
echo.
echo Field Mapper stopped with an error - scroll up to see what went wrong.
pause
exit /b 1
