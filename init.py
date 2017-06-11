# -*- coding: utf-8 -*-
from flask import Flask, Response, jsonify, request, g, render_template
from flask.ext.socketio import SocketIO, emit
import os, json, time
from model import Encoder, Generator
from interpolate import interpolate
import torch
import numpy as np
from torch.autograd import Variable



app = Flask(__name__, static_path='/static')	
socketio = SocketIO(app)

g = Generator(z_dim=100)
g.eval()
g.cuda()
g.load_state_dict(torch.load('./250000-G.pkl'))

with open('./game_data.json') as file:
	data = json.load(file)
	file.close()

def to_var(x):
    x = torch.from_numpy(np.asarray(x)).float()
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def denorm(x):
    x = (x + 1) / 2
    return x.clamp(0, 1)

@app.route('/')
def form():
	return render_template('index.html', data=data)

@app.route('/generate_image')
def generate_image():
	operation = json.loads(request.args.get('operation'))
	print operation
	if operation == 0:
		z = json.loads(request.args.get('new_z'))
		z = to_var(z)
		if z.dim() == 1:
				z = z.unsqueeze(0)
		fake_image = g(z)
		fake_image = (denorm(fake_image.squeeze().data).cpu() * 255).long()
		fake_image = fake_image.numpy().transpose(1,2,0).reshape(-1).tolist()
	else:
		z1 = json.loads(request.args.get('new_z1'))
		z2 = json.loads(request.args.get('new_z2'))
		z3 = json.loads(request.args.get('new_z3'))
		z4 = json.loads(request.args.get('new_z4'))
		fake_image = interpolate(z1, z2, z3, z4)
	return json.dumps({'generated_image': fake_image})

# Execute the main program
if __name__ == '__main__':
	socketio.run(app, host='0.0.0.0', port=6700, debug=True)