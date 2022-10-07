rem set PYTHON_DIR=C:\Tools\Python37
rem set PYTHON_DIR=D:\Tools\Anaconda42_x64
set PYTHON_DIR=D:\Tools\anaconda3\envs\torch367

set PROJECT_DIR=..

set PYTHONPATH=%PYTHONPATH%;%PROJECT_DIR%
set PATH=%PATH%;%PYTHON_DIR%;%PYTHON_DIR%\Scripts

call make html

PAUSE