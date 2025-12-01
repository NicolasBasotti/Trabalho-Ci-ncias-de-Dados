@echo off
echo Instalando dependencias para o projeto Streamlit...
echo -----------------------------------------------------

pip install --upgrade pip

pip install streamlit
pip install pandas
pip install numpy
pip install requests
pip install scikit-learn
pip install seaborn
pip install matplotlib
pip install joblib

echo.
echo -----------------------------------------------------
echo Todas as dependencias foram instaladas com sucesso!
echo Pressione qualquer tecla para fechar...
pause >nul
