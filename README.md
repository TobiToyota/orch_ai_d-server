# orch_ai_d-server

Webapp: orchaid.de

Android App: https://play.google.com/store/apps/details?id=ai.tobitoyota.orchaid

-----------------


models : 
  - additional files for "BoundingBoxes_without_save_relative.py"

utils :
  - additional files for "BoundingBoxes_without_save_relative.py"

weights :
  - saved weights for the Neural Networks

BoundingBoxes_without_save_relative.py :
  - Base file for the BoundingBoxes class
  - Additional information on the postion of the boxes for "working_relatives.py"

CL_orchid_type.py :
  - Base file for the type classification

CL_sicknesses_RL.py :
  - Base file for the sickness classifiation
  - Takes in weightings depending on the importance of the class and orchid part

database.json :
  - Instructions on how to help the orchid
  - Format so the "text_generator.py" can generate usefull texts

orch_ai_d.py :
  - Script which combines all the AIs
  - Returns the output of the system 

server.py :
  - Main server file
  - Server calls using Flask API

start_server.sh :
  - Shell script for starting the server

text_generator.py :
  - Script for generationg the output text based on the output of the system

working_relatives.py :
  - Script for sorting out the non-relevant boxes from the BoundingBoxes
