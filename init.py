# -*- coding: utf-8 -*-
from flask import Flask, Response, jsonify, request, g, render_template
from flask.ext.socketio import SocketIO, emit
import os, json, time

app = Flask(__name__, static_path='/static')	
socketio = SocketIO(app)

with open('./t10k.json') as file:
	data = json.load(file)
	data = data[0:10]
	print len(data)
	file.close()

@app.route('/')
def form():
	return render_template('index.html', data=data)

# Execute the main program
if __name__ == '__main__':
	print len(data)
	socketio.run(app, host='0.0.0.0', port=2300, debug=True)