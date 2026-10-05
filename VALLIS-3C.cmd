@echo off
setlocal EnableExtensions
pushd "%~dp0"
if errorlevel 1 (
    echo Could not open the VALLIS-3C folder.
    pause
    exit /b 1
)

if not defined LOCALAPPDATA (
    echo The Windows LOCALAPPDATA folder is unavailable.
    pause
    popd
    exit /b 1
)

set "VENV_DIR=%LOCALAPPDATA%\VALLIS-3C\1.6.0\venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
set "PYTHONDONTWRITEBYTECODE=1"

if not exist "%PYTHON_EXE%" (
    echo The shared VALLIS-3C environment has not been installed.
    echo Run INSTALL_VALLIS.cmd once, then start VALLIS-3C.cmd again.
    pause
    popd
    exit /b 1
)

"%PYTHON_EXE%" "%~dp0tools\check_environment.py" --quiet >nul 2>&1
if errorlevel 1 (
    echo The private VALLIS-3C environment is incomplete.
    echo Run INSTALL_VALLIS.cmd to verify or repair it, then try again.
    pause
    popd
    exit /b 1
)

"%PYTHON_EXE%" -m app
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo VALLIS-3C could not start or terminated with an error. Exit code: %EXIT_CODE%
    pause
)
popd
exit /b %EXIT_CODE%
