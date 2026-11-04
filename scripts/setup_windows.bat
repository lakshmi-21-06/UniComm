@echo off
cd /d "%~dp0\.."
py -3.11 -m venv .venv
if errorlevel 1 py -3.10 -m venv .venv
if errorlevel 1 (echo Install Python 3.10 or 3.11 first.&pause&exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
if errorlevel 1 (echo Backend install failed.&pause&exit /b 1)
if not exist backend\.env copy backend\.env.example backend\.env
cd frontend
npm install
if errorlevel 1 (echo Install Node.js 20+ then retry.&pause&exit /b 1)
echo Setup finished. Configure backend\.env and start both services.
pause
