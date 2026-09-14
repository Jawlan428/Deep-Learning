@echo off
REM ---------------------------------------------------------------------
REM run_verify.bat — double-click this, no typing needed.
REM
REM It works out its own location, so it runs correctly no matter where
REM you launch it from. %~dp0 is the folder this .bat lives in (cloud\),
REM always with a trailing backslash.
REM
REM   %~dp0..            -> project root (Deep-Learning-main)
REM   %~dp0..\..\.venv   -> the virtualenv, which sits one level ABOVE
REM                         the code folder
REM
REM Optional: to also run the accuracy check on your validation set,
REM drag the classifier_dataset\val folder onto this .bat file.
REM ---------------------------------------------------------------------

setlocal
set "PY=%~dp0..\..\.venv\Scripts\python.exe"
set "LOG=%~dp0models\verify_output.txt"

cd /d "%~dp0.."

if not exist "%PY%" (
    echo.
    echo ERROR: could not find the virtualenv Python at:
    echo   %PY%
    echo.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo Running Part A only ^(benchmark forensics^).
    echo To include the accuracy check, drag your classifier_dataset\val
    echo folder onto this .bat file and it will run Part B too.
    echo.
    "%PY%" cloud\verify_stage1.py > "%LOG%" 2>&1
) else (
    echo Running Part A + Part B with dataset: %~1
    echo.
    "%PY%" cloud\verify_stage1.py --dataset "%~1" > "%LOG%" 2>&1
)

REM Print the captured output to the window so you can read it here,
REM and leave a copy on disk that is easy to copy-paste back to Claude.
type "%LOG%"

echo.
echo ---------------------------------------------------------------------
echo Output also saved to:
echo   %LOG%
echo ---------------------------------------------------------------------
echo.
pause
