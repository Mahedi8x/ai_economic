@echo off
title Cloudflare Live Public Tunnel for Economic Dashboard
cd /d "C:\ai_economic"
echo ==================================================================
echo  Starting Cloudflare Live Public Tunnel for Streamlit Dashboard...
echo ==================================================================
echo.
echo Your dashboard on port 8501 is being forwarded to a public HTTPS URL.
echo.
cloudflared.exe tunnel --url http://localhost:8501
pause
