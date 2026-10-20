@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"
if errorlevel 1 (
    echo [오류] 프로젝트 폴더로 이동할 수 없습니다.
    echo 스크립트 위치와 폴더 권한을 확인하세요.
    pause
    exit /b 1
)

set "PYTHON=%CD%\.venv\Scripts\python.exe"
set "APP=%CD%\src\lotto_pattern_lab\app.py"

if not exist "%PYTHON%" (
    echo [오류] 프로젝트 가상환경을 찾을 수 없습니다.
    echo 필요한 파일: %PYTHON%
    echo 기존 .venv 가상환경을 복구한 뒤 다시 실행하세요.
    pause
    exit /b 1
)

if not exist "%APP%" (
    echo [오류] AI 패턴 연구실 실행 파일을 찾을 수 없습니다.
    echo 필요한 파일: %APP%
    pause
    exit /b 1
)

"%PYTHON%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [오류] 기존 가상환경에 Streamlit이 설치되어 있지 않습니다.
    echo 다음 명령으로 필수 패키지를 설치한 뒤 다시 실행하세요:
    echo .venv\Scripts\python.exe -m pip install -e .
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\check_streamlit_server.ps1" -ProjectRoot "%CD%" -Port 8501
set "SERVER_GUARD_EXIT=%ERRORLEVEL%"
if "%SERVER_GUARD_EXIT%"=="0" goto already_running
if "%SERVER_GUARD_EXIT%"=="1" goto port_conflict
if not "%SERVER_GUARD_EXIT%"=="2" goto server_check_failed
goto start_server

:already_running
echo 기존 AI 패턴 연구실 화면을 엽니다: http://localhost:8501
start "" "http://localhost:8501"
echo 같은 프로젝트의 기존 서버를 유지합니다. 이 창은 키를 누르면 닫힙니다.
pause >nul
exit /b 0

:port_conflict
echo 8501 포트를 다른 프로젝트 또는 프로그램이 사용 중입니다.
echo 해당 프로세스를 종료하지 않았습니다. 위의 실행 경로를 확인하세요.
pause >nul
exit /b 1

:server_check_failed
echo 8501 포트 상태를 확인하지 못했습니다. 기존 서버를 종료하거나 새 서버를 시작하지 않았습니다.
pause >nul
exit /b 1

:start_server

echo AI 패턴 연구실을 시작합니다. 서버를 종료하려면 Ctrl+C를 누르세요.
"%PYTHON%" -m streamlit run "%APP%" --server.port 8501 --server.headless false
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [오류] AI 패턴 연구실 실행이 종료되었습니다. 종료 코드: %EXIT_CODE%
)
echo.
echo 터미널을 닫으려면 아무 키나 누르세요.
pause >nul
exit /b %EXIT_CODE%
