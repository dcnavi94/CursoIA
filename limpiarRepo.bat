@echo off
echo 🚨 ESTE PROCESO BORRARÁ TU HISTORIAL LOCAL DE GIT
pause

:: Eliminar configuración Git existente
rd /s /q .git

:: Re-inicializar el repositorio
git init
git remote add origin https://github.com/dcnavi94/CursoIA.git

:: Añadir y commitear todos los archivos actuales
git add .
git commit -m "Repositorio limpio sin venv ni caché"

:: Forzar el push al repositorio remoto
git branch -M main
git push --force origin main

echo ✅ Repositorio reiniciado correctamente.
pause

