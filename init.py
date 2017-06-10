# -*- coding: utf-8 -*-
from flask import Flask, Response, jsonify, request, g, render_template
from flask.ext.socketio import SocketIO, emit
import os, json, time
from model import E, D, G
import torch
import numpy as np
from torch.autograd import Variable


app = Flask(__name__, static_path='/static')	
socketio = SocketIO(app)

g = G(z_dim=25)
g.cuda()
g.load_state_dict(torch.load('./20000_g.pkl'))

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
	z = json.load(request.arg.get('new_z'))
	z = to_var(z)
	if z.dim() == 1:
			z = z.unsqueeze(0)
	fake_image = g(z)
	fake_image = (denorm(fake_image.squeeze().data).cpu() * 255).long()
	fake_image = fake_image.numpy().reshape(-1).tolist()
	return json.dump({'generated_image': fake_image})

# Execute the main program
if __name__ == '__main__':
	print len(data)
	socketio.run(app, host='0.0.0.0', port=8600, debug=True)