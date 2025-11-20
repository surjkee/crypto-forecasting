@echo off
cd /d D:\Projects\crypto-forecasting
call .venv\Scripts\activate.bat
python -m jobs.hourly_update
