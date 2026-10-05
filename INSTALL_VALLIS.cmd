@echo off
setlocal EnableExtensions

if /I "%~1"=="--help" goto :help
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"

pushd "%~dp0"
if errorlevel 1 (
    echo ERROR: Could not open the VALLIS-3C folder.
    exit /b 1
)

if not defined LOCALAPPDATA (
    echo ERROR: The Windows LOCALAPPDATA folder is unavailable.
    goto :failure
)

set "VENV_DIR=%LOCALAPPDATA%\VALLIS-3C\1.6.0\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "BOOTSTRAP_PYTHON="
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_NO_CACHE_DIR=1"
set "PYTHONDONTWRITEBYTECODE=1"

echo.
echo VALLIS-3C dependency installer
echo ==============================

if exist "%VENV_PYTHON%" goto :check_venv

call :find_supported_python
if defined BOOTSTRAP_PYTHON goto :create_venv

echo ERROR: A supported 64-bit CPython installation was not found.
echo Install Python 3.12, 3.13 or 3.14 from:
echo https://www.python.org/downloads/windows/
echo Enable the Python launcher or Add Python to PATH, then run this installer again.
goto :failure

:create_venv
echo Creating the shared private environment in:
echo %VENV_DIR%
%BOOTSTRAP_PYTHON% -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo ERROR: The virtual environment could not be created.
    goto :failure
)

:check_venv
"%VENV_PYTHON%" -c "import struct, sys; v=sys.version_info[:2]; raise SystemExit(0 if sys.implementation.name == 'cpython' and v in ((3, 12), (3, 13), (3, 14)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: The shared environment does not use supported 64-bit CPython 3.12-3.14.
    echo Remove the environment shown below and run INSTALL_VALLIS.cmd again:
    echo %VENV_DIR%
    goto :failure
)

"%VENV_PYTHON%" "%~dp0tools\check_environment.py" --quiet >nul 2>&1
if not errorlevel 1 goto :validate_environment

echo Installing the exact VALLIS-3C dependencies ...
"%VENV_PYTHON%" -m pip install --disable-pip-version-check --no-cache-dir --require-virtualenv --only-binary=:all: -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo ERROR: One or more dependencies could not be installed.
    goto :failure
)

:validate_environment
echo Verifying installed package dependencies ...
"%VENV_PYTHON%" -m pip check
if errorlevel 1 (
    echo ERROR: pip reported an inconsistent environment.
    goto :failure
)

"%VENV_PYTHON%" "%~dp0tools\check_environment.py"
if errorlevel 1 (
    echo ERROR: The private environment is incomplete.
    goto :failure
)

echo Running the VALLIS-3C numerical validation ...
"%VENV_PYTHON%" "%~dp0tools\validate_installation.py"
if errorlevel 1 (
    echo ERROR: The numerical validation failed.
    goto :failure
)

echo.
echo Installation completed successfully.
echo Start VALLIS-3C with VALLIS-3C.cmd.
popd
if not defined NO_PAUSE pause
exit /b 0

:failure
echo.
echo Installation did not complete.
popd
if not defined NO_PAUSE pause
exit /b 1

:help
echo Usage: INSTALL_VALLIS.cmd [--no-pause]
echo.
echo Uses an installed 64-bit CPython 3.12, 3.13 or 3.14 and creates one shared
echo private environment in LOCALAPPDATA. It installs the exact packages in
echo requirements.txt without retaining a pip download cache, checks the
echo environment and runs the packaged numerical validation.
exit /b 0

:find_supported_python
set "BOOTSTRAP_PYTHON="

where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import struct, sys; raise SystemExit(0 if sys.implementation.name == 'cpython' and struct.calcsize('P') == 8 else 1)" >nul 2>&1
    if not errorlevel 1 set "BOOTSTRAP_PYTHON=py -3.12"
)
if defined BOOTSTRAP_PYTHON exit /b 0

where py >nul 2>&1
if not errorlevel 1 (
    py -3.13 -c "import struct, sys; raise SystemExit(0 if sys.implementation.name == 'cpython' and struct.calcsize('P') == 8 else 1)" >nul 2>&1
    if not errorlevel 1 set "BOOTSTRAP_PYTHON=py -3.13"
)
if defined BOOTSTRAP_PYTHON exit /b 0

where py >nul 2>&1
if not errorlevel 1 (
    py -3.14 -c "import struct, sys; raise SystemExit(0 if sys.implementation.name == 'cpython' and struct.calcsize('P') == 8 else 1)" >nul 2>&1
    if not errorlevel 1 set "BOOTSTRAP_PYTHON=py -3.14"
)
if defined BOOTSTRAP_PYTHON exit /b 0

for %%V in (312 313 314) do (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" (
        "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" -c "import struct, sys; v=sys.version_info[:2]; raise SystemExit(0 if sys.implementation.name == 'cpython' and v in ((3, 12), (3, 13), (3, 14)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
        if not errorlevel 1 set "BOOTSTRAP_PYTHON="%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe""
    )
    if defined BOOTSTRAP_PYTHON exit /b 0
)

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import struct, sys; v=sys.version_info[:2]; raise SystemExit(0 if sys.implementation.name == 'cpython' and v in ((3, 12), (3, 13), (3, 14)) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
    if not errorlevel 1 set "BOOTSTRAP_PYTHON=python"
)
exit /b 0
