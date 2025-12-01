@echo off
echo Iniciando o aplicativo Streamlit...
echo Certifique-se de que o arquivo app.py está na mesma pasta deste .bat
echo ------------------------------------------------------

REM Caminho do Python (automático se estiver no PATH)
python -m streamlit run app.py

echo.
echo ------------------------------------------------------
echo O Streamlit foi encerrado.
echo Pressione qualquer tecla para fechar...
pause >nul
