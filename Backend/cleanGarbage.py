import os
import json
pathImgPool = "/clipdraw/StorySketch_backend/static/ImgPool"
pathSketchPool = "/clipdraw/StorySketch_backend/static/SketchPool"
pathDatabaseFile = "/clipdraw/StorySketch_backend/static/sketchDatabasePool/bopan_sketchDatabase.json"
with open(pathDatabaseFile, 'r') as sketchDatabase_file:
    sketchDatabase = json.load(sketchDatabase_file) 
exsitingSketch = []
for objName in sketchDatabase:
    exsitingSketch +=  sketchDatabase[objName]
print("exsitingSketch:\n" + str(exsitingSketch))
#delete useless img
for fileName in os.listdir(pathImgPool):
    if fileName[0] == '.' or fileName=="dummy.txt":
        continue
    sketchName = fileName.split(".")[0]
    if sketchName not in exsitingSketch:
        os.remove(pathImgPool+"/"+fileName)
#delete useless sketch
for fileName in os.listdir(pathSketchPool):
    if fileName[0] == '.' or fileName=="dummy.txt":
        continue
    sketchName = fileName.split(".")[0]
    if sketchName not in exsitingSketch:
        os.remove(pathSketchPool+"/"+fileName)