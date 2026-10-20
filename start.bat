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

netstat -ano -p tcp | findstr /R /C:":8501 .*LISTENING" >nul
if not errorlevel 1 (
    echo Streamlit이 이미 8501 포트에서 실행 중입니다.
    echo 기존 AI 패턴 연구실 화면을 엽니다: http://localhost:8501
    start "" "http://localhost:8501"
    echo 기존 실행을 유지합니다. 이 창은 키를 누르면 닫힙니다.
    pause >nul
    exit /b 0
)

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
