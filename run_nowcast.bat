@echo off
cd /d "C:\ai_economic"
"c:\Users\Mehedi\anaconda4\envs\tf-gpu\python.exe" "C:\ai_economic\scripts\daily_realtime_nowcast_update.py" >> "C:\ai_economic\nowcast_cron.log" 2>&1
