@echo off
REM ---------------------------------------------------------------------
REM run_api.bat — start the LFT Reader API locally. Just double-click it.
REM
REM Installs the three web packages if they are missing, starts uvicorn,
REM and opens your browser at the upload page.
REM
REM To stop the server: click this window and press Ctrl+C.
REM ---------------------------------------------------------------------

setlocal
set "PY=%~dp0..\..\.venv\Scripts\python.exe"

cd /d "%~dp0"

if not exist "%PY%" (
    echo.
    echo ERROR: virtualenv Python not found at:
    echo   %PY%
    echo.
    pause
    exit /b 1
)

echo ===============================================================
echo  Checking web dependencies
echo ===============================================================
REM Import-test rather than always installing: pip is slow and noisy when
REM everything is already present.
"%PY%" -c "import fastapi, uvicorn, multipart" >nul 2>&1
if errorlevel 1 (
    echo  Installing fastapi, uvicorn, python-multipart ...
    "%PY%" -m pip install fastapi "uvicorn[standard]" python-multipart
    if errorlevel 1 (
        echo.
        echo  Install failed. Check your internet connection.
        pause
        exit /b 1
    )
) else (
    echo  Already installed.
)

echo.
echo ===============================================================
echo  Checking models
echo ===============================================================
if not exist "%~dp0models\detector.onnx" (
    echo  ERROR: models\detector.onnx missing. Run cloud\export_onnx.py first.
    pause
    exit /b 1
)
if not exist "%~dp0models\classifier.onnx" (
    echo  ERROR: models\classifier.onnx missing. Run cloud\export_onnx.py first.
    pause
    exit /b 1
)
echo  detector.onnx   OK
echo  classifier.onnx OK

echo.
echo ===============================================================
echo  Starting server on http://127.0.0.1:8080
echo ===============================================================
echo.
echo   Upload page : http://127.0.0.1:8080
echo   API docs    : http://127.0.0.1:8080/docs
echo   Health      : http://127.0.0.1:8080/health
echo.
echo   Press Ctrl+C in this window to stop.
echo.

REM Open the browser after a short delay so the server is up first.
start "" /b cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8080"

REM --reload restarts the server whenever a .py file changes. Handy while
REM developing; it is NOT used in the container, where the code is fixed.
"%PY%" -m uvicorn main:app --reload --port 8080

echo.
echo Server stopped.
pause
