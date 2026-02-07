@echo off
REM run_all_configs_simple.bat
REM Simple script to run all configs sequentially on Windows
REM
REM Usage:
REM   run_all_configs_simple.bat                    # Uses config\ directory (default)
REM   run_all_configs_simple.bat config\1202        # Uses specific directory
REM   run_all_configs_simple.bat config\experiments # Uses custom directory

setlocal enabledelayedexpansion

REM Get config directory from argument or use default
if "%~1"=="" (
    set "CONFIG_DIR=config"
) else (
    set "CONFIG_DIR=%~1"
)

REM Check if directory exists
if not exist "%CONFIG_DIR%\" (
    echo Error: Directory '%CONFIG_DIR%' does not exist
    echo Usage: %~nx0 [config_directory]
    echo Example: %~nx0 config\1202
    exit /b 1
)

REM Count yaml files
set YAML_COUNT=0
for %%f in ("%CONFIG_DIR%\*.yaml") do set /a YAML_COUNT+=1

if %YAML_COUNT% equ 0 (
    echo Error: No .yaml files found in '%CONFIG_DIR%'
    exit /b 1
)

echo ========================================================================
echo Running all configs from: %CONFIG_DIR%
echo Found %YAML_COUNT% config file(s)
echo ========================================================================
echo.

REM Activate conda environment
call conda activate sml

REM Run each config file
set CONFIG_NUM=0
for %%f in ("%CONFIG_DIR%\*.yaml") do (
    set /a CONFIG_NUM+=1
    echo [!CONFIG_NUM!/%YAML_COUNT%] Running: %%f
    echo ------------------------------------------------------------------------

    python main.py --configs "%%f"
    set EXIT_CODE=!errorlevel!

    if !EXIT_CODE! equ 0 (
        echo [OK] Completed: %%f
    ) else (
        echo [FAIL] Failed: %%f (exit code: !EXIT_CODE!^)
    )

    echo ------------------------------------------------------------------------
    echo.
)

echo ========================================================================
echo All configs completed!
echo Total: %CONFIG_NUM% config(s) processed from %CONFIG_DIR%
echo ========================================================================

endlocal
