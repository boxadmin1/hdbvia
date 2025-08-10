@echo off
title HDBVIA - Human Detection ^& Biometrics Verification and Identification Algorithm

echo -----------------------------------------------
echo Project: Human Detection ^& Biometrics Verification and Identification Algorithm (HDBVIA)
echo Created by Evan O
echo Warning! In 5 seconds this program will launch hdbvia.py and install Python libraries.
echo This program accesses your camera and runs other tasks.
echo If you do not want this, close the window and delete this file and hdbvia.py.
echo -----------------------------------------------
timeout /t 5 /nobreak >nul

echo Installing required Python packages...
python -m pip install --user mediapipe opencv-python

echo Starting HDBVIA...
python hdbvia.py

pause
