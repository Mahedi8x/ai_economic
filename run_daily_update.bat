@echo off
title Daily Economic Intelligence Update
cd /d "C:\ai_economic"
echo ==================================================================
echo  Running Daily Live Update & AI Forecasting Engine
echo  Timestamp: %date% %time%
echo ==================================================================
python daily_update.py
echo.
timeout /t 5 >nul
