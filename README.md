# NOTE: main.py is not currently being worked on. It's only used as a reference before we switch over to mediapipe fully 

# Driver Drowsiness Awareness System
This project is a driver drowsiness awareness system. Developed for the Samsung Innovation Campus capstone project.

Requirements are basic, as only a PC with a webcam is needed to run the application. The application is built using the MediaPipe model, which is a Google Tensorflow model that is trained on the MediaPipe dataset. It's much faster than dlib, and much less resource intensive compared to dlib/cnn. 

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
