@echo off
setlocal

:: Create SSL directory if it does not exist
set "DIR_SSL=%~dp0settings\ssl"
if not exist "%DIR_SSL%" (
    echo Creating directory: %DIR_SSL%
    mkdir "%DIR_SSL%"
)

:: Set file paths
set "FILE_SSL_CERT_PEM=%DIR_SSL%\cert.pem"
set "FILE_SSL_KEY_PEM=%DIR_SSL%\key.pem"

:: Check for OpenSSL
set "OPENSSL_BIN=C:\Program Files\OpenSSL-Win64\bin\openssl.exe"

if not exist "%OPENSSL_BIN%" (
    echo ERROR: OpenSSL binary not found at expected path.
    pause
    exit /b 1
)

:: Generate cert if missing
if not exist "%FILE_SSL_CERT_PEM%" (
    echo Generating SSL certificate...
    "%OPENSSL_BIN%" req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes ^
    -keyout "%FILE_SSL_KEY_PEM%" -out "%FILE_SSL_CERT_PEM%" ^
    -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost"
)

:: Launch the server
python controller\app.py settings

endlocal
