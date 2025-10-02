

# deactivate any active virtual environment
deactivate

# Get the current script location and set it as the working directory
$location = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $location

# change to the ..\sck-core-framework folder
Set-Location ..\sck-core-framework

# activate virtual environment
& .\.venv\Scripts\Activate.ps1

# dump the requirements to a file using the uv command
uv export -f requirements.txt --without-hashes --output ..\sck-core-ai\build_requirements.txt

# deactivate the virtual environment
deactivate

# return to the original location
Set-Location $location

