@echo off
REM Double-click to run. First run sets up Python packages (a few minutes).
cd /d "%~dp0"
if not exist .venv (
    echo Setting up for the first time...
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)
python run.py %*
pause
