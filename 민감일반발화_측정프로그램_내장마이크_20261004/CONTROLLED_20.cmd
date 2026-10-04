@echo off
setlocal
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
chcp 65001 >nul
cd /d "%~dp0program"
"%~dp0runtime\python.exe" -s launch.py --controlled
if errorlevel 1 pause
