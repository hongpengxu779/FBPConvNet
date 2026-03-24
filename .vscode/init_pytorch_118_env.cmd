@echo off
CALL C:\ProgramData\anaconda3\Scripts\activate.bat C:\ProgramData\anaconda3\envs\pytorch_118
IF ERRORLEVEL 1 (
    echo Failed to activate conda environment: pytorch_118
) ELSE (
    echo Activated conda environment: pytorch_118
)
