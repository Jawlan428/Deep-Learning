@echo off
REM ---------------------------------------------------------------------
REM deploy_gcp.bat — push the tested image to Artifact Registry and deploy
REM to Cloud Run.
REM
REM RUN THIS ONLY AFTER steps 1-4 of DEPLOY.md:
REM   1. gcloud CLI installed
REM   2. Google Cloud account + project created
REM   3. $1 budget alert set          <-- do not skip
REM   4. APIs enabled, lft-repo created
REM
REM It deploys the image you ALREADY built and verified locally, rather
REM than rebuilding from source in the cloud. A cloud rebuild is a
REM different artifact that you have not tested.
REM ---------------------------------------------------------------------

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set REGION=us-central1
set REPO=lft-repo
set IMAGE=lft-reader
set TAG=v1
set SERVICE=lft-reader

echo ===============================================================
echo  LFT Reader - deploy to Cloud Run
echo ===============================================================
echo.

REM ---- preflight -------------------------------------------------
where gcloud >nul 2>&1
if errorlevel 1 (
    echo  ERROR: gcloud not found on PATH.
    echo  Install the Google Cloud CLI, then open a NEW terminal window
    echo  so it picks up the updated PATH.
    echo    https://cloud.google.com/sdk/docs/install
    pause
    exit /b 1
)

docker version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Docker is not responding. Start Docker Desktop.
    pause
    exit /b 1
)

docker image inspect %IMAGE%:%TAG% >nul 2>&1
if errorlevel 1 (
    echo  ERROR: local image %IMAGE%:%TAG% not found.
    echo  Run run_docker.bat first to build and test it.
    pause
    exit /b 1
)

echo  gcloud  OK
echo  docker  OK
echo  image   %IMAGE%:%TAG% present
echo.

REM ---- project ---------------------------------------------------
REM The project created on 14/09/2026. Overridden by whatever gcloud is
REM already configured with, so this is only a convenience default.
set DEFAULT_PROJECT=project-512fcc90-021f-4da9-a2a

for /f "tokens=*" %%i in ('gcloud config get-value project 2^>nul') do set PROJECT=%%i

if "%PROJECT%"=="" (
    echo  No project configured in gcloud. Using the known project:
    echo    %DEFAULT_PROJECT%
    echo.
    set PROJECT=%DEFAULT_PROJECT%
    call gcloud config set project %DEFAULT_PROJECT%
)

echo  project : %PROJECT%
echo  region  : %REGION%
echo.

echo ===============================================================
echo  CONFIRM BEFORE CONTINUING
echo ===============================================================
echo.
echo  Have you set the $1 budget alert? (DEPLOY.md step 3)
echo  This should never fire - which is exactly why it is worth having.
echo.
set /p BUDGET="  Type YES to continue: "
if /i not "%BUDGET%"=="YES" (
    echo.
    echo  Stopping. Set the budget alert first, then run this again.
    pause
    exit /b 0
)

set TARGET=%REGION%-docker.pkg.dev/%PROJECT%/%REPO%/%IMAGE%:%TAG%

echo.
echo ===============================================================
echo  1/3  Authenticating Docker to Artifact Registry
echo ===============================================================
REM Writes a credential helper into Docker's config. No password is
REM stored - it calls gcloud for a short-lived token on each push.
call gcloud auth configure-docker %REGION%-docker.pkg.dev --quiet
if errorlevel 1 goto :failed

echo.
echo ===============================================================
echo  2/3  Tagging and pushing
echo ===============================================================
echo  target: %TARGET%
echo.
echo  A Docker tag is a NAME, not a copy - tagging is instant. The push
echo  uploads roughly 150-200 MB compressed and takes a few minutes on
echo  a normal connection.
echo.

docker tag %IMAGE%:%TAG% %TARGET%
if errorlevel 1 goto :failed

docker push %TARGET%
if errorlevel 1 goto :failed

echo.
echo ===============================================================
echo  3/3  Deploying to Cloud Run
echo ===============================================================
echo.

call gcloud run deploy %SERVICE% ^
    --image=%TARGET% ^
    --region=%REGION% ^
    --platform=managed ^
    --memory=1Gi ^
    --cpu=2 ^
    --min-instances=0 ^
    --max-instances=3 ^
    --concurrency=4 ^
    --timeout=60 ^
    --allow-unauthenticated ^
    --quiet
if errorlevel 1 goto :failed

echo.
echo ===============================================================
echo  DEPLOYED
echo ===============================================================
for /f "tokens=*" %%u in ('gcloud run services describe %SERVICE% --region %REGION% --format "value(status.url)"') do set URL=%%u
echo.
echo   Live URL : %URL%
echo   Health   : %URL%/health
echo   API docs : %URL%/docs
echo.
echo   Test it:
echo     curl %URL%/health
echo.
echo   Wait a few minutes, THEN time the first request. That is your real
echo   cold start - the number worth quoting, not the 373 ms from local.
echo.
start "" %URL%
pause
exit /b 0

:failed
echo.
echo ===============================================================
echo  FAILED - the error is above
echo ===============================================================
echo.
echo  Common causes:
echo    - APIs not enabled:
echo        gcloud services enable artifactregistry.googleapis.com run.googleapis.com
echo    - repository missing:
echo        gcloud artifacts repositories create %REPO% --repository-format=docker --location=%REGION%
echo    - billing not linked to the project (check the Console)
echo    - wrong project ID (display name is not the same as the ID)
echo.
pause
exit /b 1
