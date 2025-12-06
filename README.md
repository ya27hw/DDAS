# NOTE: main.py is not currently being worked on. It's only used as a reference before we switch over to mediapipe fully 

# Driver Drowsiness Awareness System
This project is a driver drowsiness awareness system. Developed for the Samsung Innovation Campus capstone project.

Current contributors:
- [Yousuf Ashraf](https://github.com/ya27hw)

# Installation Procedure
## Step 1
Create a python venv
```shell
python3 -m venv venv
```
Activate the venv
```shell
source venv/bin/activate
```
Or on Windows:
```shell
venv\Scripts\activate.bat
```

`Note: Install CMake tools. If on Windows, you need to run the Visual Studio installer. If on Linux, see instructions for your specific distro.`

## Step 2
Install requirements
```shell
pip install -r requirements.txt
```

## Step 3
Run the application
```shell
python mp.py
```

# Credits
Dlib for face detection and tracking