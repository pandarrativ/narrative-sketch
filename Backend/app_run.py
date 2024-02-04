import pprint 
import sys  
import copy
from time import time
from typing import Type
# sys.path.append("../")
from storySketch import User
import os
import json
from flask import Flask, jsonify, request, Blueprint, render_template, abort, send_from_directory



GPT_url = 'https://hz-t3.matpool.com:28238/continuePrompt'

# Initialize the app
app = Flask(__name__)

def requestParse(req_data):
    """解析请求数据并以json形式返回"""
    if req_data.method == "POST":
        if req_data.json != None:
            data = req_data.json
        else:
            data = req_data.form
    elif req_data.method == "GET":
        data = req_data.args
    return data

# Initialize nl4dv variable
user_instance = None

@app.route('/init', methods=['POST'])
# @cross_origin()
def init():
    global user_instance
    if user_instance is not None:
        return jsonify({"message":"user is already initialized"})
    userName = requestParse(request)['userName']
    user_instance = User(userName, request.host_url, GPT_url)
    return jsonify({"message":"Story sketch is initialized, user is "+userName})

@app.route('/GetStateName', methods=['POST'])
# @cross_origin()
def GetStateName():
    global user_instance
    if user_instance is None:
        return jsonify({"message":"Fail: no user"})
    if user_instance.curState is not None:
        return jsonify({"StateName":user_instance.curState.stateName})
    else:
        return jsonify({"StateName":"NoneState"})

@app.route('/newState', methods=['POST'])
# @cross_origin()
def newState():
    global user_instance
    if user_instance is None:
        return jsonify({"message":"Fail: no user"})
    parsedRequest = requestParse(request)
    stateType = parsedRequest['stateType']
    prevStateName = parsedRequest['prevStateName']

    if user_instance.curState is not None:
        return jsonify({"message":"Fail: you are already in state: " + user_instance.curState.stateName})
    else:
        if stateType == "Combination":
            result = user_instance.newState(stateType, prevStateName, parsedRequest["endStateNameList"])
        else:
            result = user_instance.newState(stateType, prevStateName)
        return jsonify( result)

@app.route('/PrepGetObjs', methods=['POST'])
# @cross_origin()
def PrepGetObjs():
    global user_instance
    if user_instance is None:
        return jsonify({"message":"Fail: no user"})
    objName = requestParse(request)['objName']
    if user_instance.curState is not None and user_instance.curState.stateType == "Preparation":
        return jsonify(user_instance.curState.PrepGetObjs(objName))
    else:
        return jsonify({"message":"Fail: not in Preparation state"})

@app.route('/ExpRefreshDirection', methods=['POST'])
# @cross_origin()
def ExpRefreshDirection():
    global user_instance
    if user_instance is None:
        return jsonify({"message":"Fail: no user"})
    directionName = requestParse(request)['directionName']
    if user_instance.curState is not None and user_instance.curState.stateType == "Exploration":
        return jsonify(user_instance.curState.ExpRefreshDirection(directionName))
    else:
        return jsonify({"message":"Fail: not in Exploration state"})

@app.route('/CombRefreshDirection', methods=['POST'])
# @cross_origin()
def CombRefreshDirection():
    global user_instance
    if user_instance is None:
        return jsonify({"message":"Fail: no user"})
    stateName = requestParse(request)['stateName']
    if user_instance.curState is not None and user_instance.curState.stateType == "Combination":
        return jsonify(user_instance.curState.CombRefreshDirection(stateName))
    else:
        return jsonify({"message":"Fail: not in Transformation state"})

@app.route('/TransRefreshDirection', methods=['POST'])
# @cross_origin()
def TransRefreshDirection():
    global user_instance
    if user_instance is None:
        return jsonify({"message":"Fail: no user"})
    directionName = requestParse(request)['directionName']
    if user_instance.curState is not None and user_instance.curState.stateType == "Transformation":
        return jsonify(user_instance.curState.TransRefreshDirection(directionName))
    else:
        return jsonify({"message":"Fail: not in Transformation state"})


@app.route('/doneState', methods=['POST'])
# @cross_origin()
def doneState():
    global user_instance
    if user_instance is None:
        return jsonify({"message":"Fail: no user"})
    selectSketch = requestParse(request)['selectSketch']
    print(selectSketch)
    if user_instance.curState is None:
        return jsonify({"message":"Fail: not in any state"})
    else:
        return jsonify(user_instance.doneState(selectSketch))

@app.route('/cancelState', methods=['POST'])
# @cross_origin()
def cancelState():
    global user_instance
    if user_instance is None:
        return jsonify({"message":"Fail: no user"})
    if user_instance.curState is None:
        return jsonify({"message":"Fail: not in any state"})
    else:
        return jsonify(user_instance.cancelState())


@app.route('/userExit', methods=['POST'])
# @cross_origin()
def userExit():
    global user_instance
    if user_instance is None:
        return jsonify({"message":"Fail: no user"})
    return jsonify(user_instance.userExit())
    
if __name__ == "__main__":

    app.run(host='0.0.0.0', debug=True, threaded=True, port=8046)
