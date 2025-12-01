@echo off
cd /d D:\Projects\crypto-forecasting
call .venv\Scripts\activate.bat
python -m jobs.train_lstm_all
python -m jobs.train_gru_all