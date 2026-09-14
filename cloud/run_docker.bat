@echo off
REM ---------------------------------------------------------------------
REM run_docker.bat — build the container and run it locally. Double-click.
REM
REM This is Stage 3. Nothing here touches the cloud and nothing costs money.
REM Get it working locally FIRST — debugging a broken container through
REM Cloud Run's logs is far slower than debugging it on your own machine.
REM
REM Requires Docker Desktop to be running (whale icon in the system tray).
REM ---------------------------------------------------------------------

setlocal
cd /d "%~dp0"

set IMAGE=lft-reader
set TAG=v1
set PORT=8081

echo ===============================================================
echo  Checking Docker
echo ===============================================================
docker version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Docker is not responding.
    echo  Start Docker Desktop and wait for the whale icon to settle.
    echo.
    pause
    exit /b 1
)
echo  Docker is running.

echo.
echo ===============================================================
echo  Building %IMAGE%:%TAG%
echo ===============================================================
echo  First build downloads the base image and compiles wheels:
echo  expect 3-6 minutes. Later builds reuse cached layers.
echo.

docker build -t %IMAGE%:%TAG% .
if errorlevel 1 (
    echo.
    echo  BUILD FAILED. The error is above.
    pause
    exit /b 1
)

echo.
echo ===============================================================
echo  Image size
echo ===============================================================
docker images %IMAGE%:%TAG% --format "  {{.Repository}}:{{.Tag}}   {{.Size}}"
echo.
echo  Target was under 500 MB. A torch-based image would be around 2 GB.

echo.
echo ===============================================================
echo  Running on http://127.0.0.1:%PORT%
echo ===============================================================
echo.
echo   --memory 1g       matches the Cloud Run limit, so if it OOMs here
echo                     it would have OOMed there
echo   --cpus 2          matches 2 vCPUs
echo   -e PORT=8080      Cloud Run injects PORT; we prove we honour it
echo   -p %PORT%:8080      host %PORT% -^> container 8080, so this does not
echo                     clash with the uvicorn dev server on 8080
echo.
echo   Press Ctrl+C to stop.
echo.

start "" /b cmd /c "timeout /t 6 >nul & start http://127.0.0.1:%PORT%"

docker run --rm -it ^
    --name %IMAGE% ^
    --memory 1g ^
    --cpus 2 ^
    -e PORT=8080 ^
    -p %PORT%:8080 ^
    %IMAGE%:%TAG%

echo.
echo Container stopped.
pause
