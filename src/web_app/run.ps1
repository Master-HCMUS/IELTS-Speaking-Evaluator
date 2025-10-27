# Run the Streamlit web application for pronunciation assessment
# This script handles OpenMP conflicts that can occur with NumPy/SciPy

Write-Host "Starting Pronunciation Assessment Web App..." -ForegroundColor Green
Write-Host ""



# Set environment variable to handle OpenMP conflicts
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Write-Host "OpenMP fix enabled" -ForegroundColor Yellow

# Run streamlit app
Write-Host "Launching Streamlit app..." -ForegroundColor Green
streamlit run src/web_app/app.py --logger.level=info
