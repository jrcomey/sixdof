import json
import numpy as np
import pretty_errors
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import random
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import onnx
from collections import deque
# torch.set_default_dtype(torch.double)
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, List
from collections import namedtuple, deque
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from analyzedata import *

# Constants
# BATCH_SIZE = 32
LR = 1E-3
RUN_NUMS = 0

class Job:

    def __init__(self, job_name="default_name", *, scenarios=[]):
        self.job_name = job_name
        self.scenarios = scenarios
        self.objects = []
        path = f"data/todo/{self.job_name}"
        self.path=path

    def add_object(self, new_object):
        duplicate_check = False

        for obj in self.objects:
            if obj.name ==new_object.name and new_object.name != "":
                duplicate_check = True

        if duplicate_check == False:
            self.objects.append(new_object)

    def add_scenario(self, new_scenario): 

        duplicate_check = False

        for obj in self.scenarios:
            if obj.scenario_name == new_scenario.scenario_name:
                duplicate_check = True

        if duplicate_check == False:
            new_scenario.job_name = self.job_name
            self.scenarios.append(new_scenario)

    def export_job(self):
        path = f"data/todo/{self.job_name}"
        self.path=path
        os.makedirs(path, exist_ok=True)
        os.makedirs(path+"/scenarios", exist_ok=True)
        os.makedirs(path+"/output", exist_ok=True)
        os.makedirs(path+"/objects", exist_ok=True)

        # print(self.scenarios)

        for scenario in self.scenarios:
            scenario.create_run_json(target_directory=path+f"/scenarios")

        for obj in self.objects:
            obj.save_sim_obj(path+f"/objects/{obj.name}")


class Scenario:

    def __init__(self, scenario_name="default_scenario_name", end_time=1.0, min_dt=1E-1, *,objects=[], environments = [], datacom_port=None):
        self.scenario_name = scenario_name
        self.end_time=end_time
        self.min_dt = min_dt
        self.objects = objects
        self.environments = environments
        self.datacom_port=datacom_port


    def add_object(self, object):
        pass

    def add_environment(self, environment):
        pass

    def create_run_json(self, *, target_directory="data/todo"):
        scenario_json = {
            "scenarioName": self.scenario_name,
            "endTime": self.end_time,
            "dtMin": self.min_dt,
            "vehicles": [obj.create_json_dict() for obj in self.objects],
            "environments": [env.create_json_dict() for env in self.environments],
            "datacomPort": self.datacom_port,
            "jobName": self.job_name.strip("/"),
        }
        # print(scenario_json)

        # Commenting out legacy code
        # # Make directories for sims if they don't already exist
        # if not os.path.isdir(input:=f"data/todo"):
        #     os.mkdir(input)
        # if not os.path.isdir(output:=f"data/runs/{self.scenario_name}"):
        #     os.mkdir(output)

        with open(target_directory+f"/{self.scenario_name}.json", "w") as file:
            file.write(json.dumps(scenario_json))


class Component:

    def __init__(self, name="", ):
        pass

    def create_json_dict():
        data = {
            "default_component": 0,
        }

        return data

    def to_json_str(self):
        return json.dumps(self.create_json_dict())


class IdealThruster(Component):

    def __init__(self,*, time_constant=0.01, position=np.array([[0.0, 0.0, 0.0]]), orientation=np.array([[0.0,0.0,0.0]])):
        self.time_constant = time_constant
        self.position = position
        self.orientation = orientation

    def create_json_dict(self):
        data= {
            "component_type": "IdealThruster",
            "time_constant": self.time_constant,
            "position": sum(self.position.tolist(),[]),
            "orientation": sum(self.orientation.tolist(),[])
        }
        return data


class FlightComputer:

    def __init__(self, sample_time, sensors, K, guidance_computer=None):

        self.sample_time = sample_time
        self.K = K

        if isinstance(self.K, np.ndarray):
            self.fcType = "StateFeedback"
        elif isinstance(self.K, nn.Module):
            self.fcType = "NeuralNet"
        else:
            self.fcType = ""

        self.guidance_computer = guidance_computer

    def set_NN_filepath(self, filepath):
        self.NN_filepath = filepath

    def update_NN(self, model):
        self.K = None
        self.K = model

    def create_json_dict(self):
        # print(self.fcType)
        if self.fcType=="NeuralNet":
            dummy_input=torch.zeros(1,12).to(torch.float32)
            model = self.K.to(torch.float32)
            exported_program: torch.export.ExportedProgram =  torch.onnx.export(
            model,
            dummy_input,
            self.NN_filepath,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            )
            data= {
                "fcType": self.fcType,
                "sample_time": self.sample_time,
            }
        else:
            data = {
            "fcType": self.fcType,
            "sample_time": self.sample_time,
            "K": sum(self.K.tolist(), [])
        }
            
        if self.guidance_computer is None:
            # data["guidanceComputer"] = "Zero"
            data["guidanceType"] = "Zero"
        elif isinstance(self.guidance_computer, TimingGuidanceComputer):
            data["guidanceType"] = "TimingGuidance"
            data["guidanceComputer"] = self.guidance_computer.create_json_dict()
        elif isinstance(self.guidance_computer, WaypointGuidanceComputer):
            data["guidanceType"] = "WaypointGuidance"
            data["guidanceComputer"] = self.guidance_computer.create_json_dict()
        return data
    
    def to_json_str(self):
        return json.dumps(self.create_json_dict())

    def add_guidance_system(self, new_guidance):
        self.guidance_computer = new_guidance


class TimingGuidanceComputer:

    def __init__(self, times=[0], states=[np.ndarray]):
        self.times = times
        self.states = states

    def create_json_dict(self):
        data = {
            "transitionTimes": self.times,
            "targetStates": [sum(state.tolist(), []) for state in self.states],
        }
        return data


class WaypointGuidanceComputer:

    def __init__(self, waypoints: [np.ndarray], tolerance: float, hold_time: float):
        self.waypoints = waypoints
        self.tolerance = tolerance
        self.hold_time = hold_time

    def create_json_dict(self):
        return {
            "waypoints": [sum(state.tolist(), []) for state in self.waypoints],
            "tolerance": self.tolerance,
            "holdTime": self.hold_time,
        }


class SimObject:

    def __init__(self, name="", mass=0.0, I = np.array([[0.0, 0.0, 0.0],[0,0,0],[0,0,0]]), *,
                 graphical_elements=[], physics_type="Static", init_state=np.zeros((12,1)),
                    dependent_components=None, B=None, fc=None, recording_sample_time = 0.0, 
                     max_recorder_buffer_steps=1000, run_name=""):
        # General purpose components
        self.name = name
        self.mass = mass
        self.I = I

        # Generate input components if any exist
        if dependent_components is not None:
            self.U = len(dependent_components)
            self.components=dependent_components
            if B is not None:
                self.B = B
            else:
                self.B=np.zeros((12, self.U))
        else:
            self.U=1
            self.components=[]
            self.B=np.zeros((12, self.U))

        self.fc = fc

        # Generate dynamic components if not static
        if physics_type != "Static":
            self.A = np.array([
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # x
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   # y
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   # z
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # x'
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # y'
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # z'
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # phi
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   # theta
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # psi
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # phi'
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # theta'
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    # psi'
                            ], dtype=float)
            
            if physics_type in ["RK4", "ForwardEuler"]:
                self.physics_type = physics_type
            else:
                self.physics_type="Static"
        else:
            self.A = np.zeros((12, 12))
            self.physics_type=physics_type

        # Initialize state
        self.state = init_state

        # Initialize graphics components
        self.graphical_elements = graphical_elements

        # Initialize data recording elements
        self.sample_time = recording_sample_time
        self.max_steps = max_recorder_buffer_steps
        self.run_name = run_name
        
    def create_json_dict(self):
        data = {
            # "name": self.name,
            "mass": self.mass,
            "rot_inertia": sum(self.I.tolist(), []),
            # "state": sum(self.state.tolist(),[]),
            "A": sum(self.A.tolist(), []),
            "B": sum(self.B.tolist(), []),
            # "physicsType": self.physics_type,
            "components": [comp.create_json_dict() for comp in self.components],
            "input_size": self.U,
            "GraphicalElements": [graph.create_json_dict() for graph in self.graphical_elements],
            "SampleTime": self.sample_time,
            "MaxSteps": self.max_steps,
            "RunName": self.run_name
        }
        if self.fc is not None:
            data["flight_controller"] = self.fc.create_json_dict()
        else:
            data["flight_controller"] = "None"
        return data

    def to_json_str(self):
        return json.dumps(self.create_json_dict())
    
    def save_sim_obj(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(directory+f"/{self.name}.json", "w") as file:
            file.write(self.to_json_str())


@dataclass
class Object_Instance:
    object_name: str
    initial_state: np.array
    physics_type: str
    input_size: int
    
    def create_json_dict(self):
        data = {
            "name": self.object_name,
            "state": sum(self.initial_state.tolist(), []),
            "physicsType": self.physics_type,
            "inputSize": self.input_size,
        }
        return data

    def to_json_str(self):
        return json.dumps(self.create_json_dict())


class Environment:

    def __init__(self):
        pass

    def create_json_dict(self):
        data = {
            "default": 0,
        }

        return data

    def to_json_str(self):
        return json.dumps(self.create_json_dict())

   
class PointMassGravity(Environment):

    def __init__(self, mass=0, soi_radius=0, position = np.array([0,0,0])):

        self.mass = mass
        self.soi_radius=soi_radius
        self.position=position

    def create_json_dict(self):
        return {
            "type": "PointMassGravity",
            "mass": self.mass,
            "soi_radius": self.soi_radius,
            "position": self.position.tolist()
        }


class ConstantField(Environment):

    def __init__(self, acceleration=9.81, direction=np.array([[0.0, 0.0, -1.0]])):
        self.acceleration = acceleration
        self.direction=direction

    def create_json_dict(self):
        return {
            "type": "ConstantField",
            "acceleration": self.acceleration,
            "direction":  sum(self.direction.tolist(),[])
        }


class GraphicalElement:

    def __init__(self, name="DEFAULT", filepath='data/test_boject/default_cube.obj', relative_position=[0,0,0], orientation=[0,0,0], rotation=[0,0,0], color=[1,0,0,1], scale=[1,1,1]):
        self.name = name
        self.filepath=filepath
        self.position=relative_position
        self.orientation=orientation
        self.rotation=rotation
        self.color=color
        self.scale=scale

    def create_json_dict(self):
        return {
            "Name": self.name,
            "ObjectFilePath": self.filepath,
            "Position": self.position,
            "Orientation": self.orientation,
            "Rotation": self.rotation,
            "Color": self.color,
            "Scale": self.scale,
        }

class BlizzardController(nn.Module):
    
    def __init__(self, input_size=12, hidden_layer_size=256, output_size=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size, dtype=torch.float32),
            nn.Linear(hidden_layer_size, hidden_layer_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, output_size, dtype=torch.float32),
            nn.Tanh(),
        )
        # print(self.network)

    def forward(self, x):
        return self.network(x)


@dataclass
class SimObjectOutput:
    time:           np.ndarray      # Time vector
    trans_true:     np.ndarray      # True translational position
    trans_cmd:      np.ndarray      # Commanded translational position
    trans_vel_true: np.ndarray      # True translational velocity
    trans_vel_cmd:  np.ndarray      # Commanded translational velocity
    rot_true:       np.ndarray      # True rotational position
    rot_cmd:        np.ndarray      # True rotational command
    rot_vel_true:   np.ndarray      # True rotational velocity
    rot_vel_cmd:    np.ndarray      # Rotational Velocity Command
    states:         np.ndarray      # Full States
    u_vector:       np.ndarray      # U

    def to_vector(self):
        states = self.states[:-1,:]
        next_states = self.states[1:,:]
        u = self.u_vector[:-1,:]
        # print(u.shape)
        return states, next_states, u
    
    def compute_acceleration(self):
        dt = np.diff(self.time)
        
        # Compute translational acceleration
        trans_accel_true = np.diff(self.trans_vel_true, axis=0) / dt[:, None]
        trans_accel_cmd = np.diff(self.trans_vel_cmd, axis=0) / dt[:, None]
        
        # Compute rotational acceleration
        rot_accel_true = np.diff(self.rot_vel_true, axis=0) / dt[:, None]
        rot_accel_cmd = np.diff(self.rot_vel_cmd, axis=0) / dt[:, None]
        
        # Compute jerk (derivative of acceleration)
        trans_jerk_true = np.diff(trans_accel_true, axis=0) / dt[:-1, None]
        trans_jerk_cmd = np.diff(trans_accel_cmd, axis=0) / dt[:-1, None]
        
        rot_jerk_true = np.diff(rot_accel_true, axis=0) / dt[:-1, None]
        rot_jerk_cmd = np.diff(rot_accel_cmd, axis=0) / dt[:-1, None]
        
        # Modified time vectors
        accel_time = self.time[:-1]  # One step shorter than time
        jerk_time = self.time[:-2]   # Two steps shorter than time
        
        return {
            "accel_time": accel_time,
            "trans_accel_true": trans_accel_true,
            "trans_accel_cmd": trans_accel_cmd,
            "rot_accel_true": rot_accel_true,
            "rot_accel_cmd": rot_accel_cmd,
            "jerk_time": jerk_time,
            "trans_jerk_true": trans_jerk_true,
            "trans_jerk_cmd": trans_jerk_cmd,
            "rot_jerk_true": rot_jerk_true,
            "rot_jerk_cmd": rot_jerk_cmd,
        }


def read_from_csv(filepath):
    df = pd.read_csv(filepath)
    time = df["Time"].to_numpy()
    full_state = df[[
        "X Position",
        "Y Position",
        "Z Position",
        "x_vel",
        "y_vel",
        "z_vel",
        "Pitch",
        "Roll",
        "Yaw",
        "pitch_vel",
        "roll_vel",
        "yaw_vel"
        ]].to_numpy()
    state_true = df[[
        "X Position",
        "Y Position",
        "Z Position",]
        # "x_vel",
        # "y_vel",
        # "z_vel"]
        # "Pitch",
        # "Roll",
        # "Yaw",
        # "pitch_vel",
        # "roll_vel",
        # "yaw_vel"
        ].to_numpy()
    
    trans_vel_true = df[[
        "x_vel",
        "y_vel",
        "z_vel"
    ]].to_numpy()

    trans_vel_cmd = df[[
        "x_vel_cmd",
        "y_vel_cmd",
        "z_vel_cmd",
    ]].to_numpy()
    
    rot_true = df[[
        "Pitch",
        "Roll",
        "Yaw",
    ]].to_numpy()

    rot_cmd = df[[
        "pitch_cmd",
        "roll_cmd",
        "yaw_cmd",
    ]].to_numpy()

    rot_vel_true = df[[
        "pitch_vel",
        "roll_vel",
        "yaw_vel"
    ]].to_numpy()

    rot_vel_cmd = df[[
        "pitch_vel_cmd",
        "roll_vel_cmd",
        "yaw_vel_cmd"
    ]].to_numpy()
    
    state_cmd = df[[
        "x_pos_cmd",
        "y_pos_cmd",
        "z_pos_cmd",]
        # "x_vel",
        # "y_vel",
        # "z_vel"]
        # "pitch",
        # "Roll",
        # "Yaw",
        # "pitch_vel",
        # "roll_vel",
        # "yaw_vel"
        ].to_numpy()
    
    u_vec = df[[
        f"U_{i}" for i in range(8)
    ]].to_numpy()
    
    return SimObjectOutput(time, state_true, state_cmd,trans_vel_true=trans_vel_true, trans_vel_cmd=trans_vel_cmd, rot_true=rot_true, rot_cmd=rot_cmd, rot_vel_true=rot_vel_true, rot_vel_cmd=rot_vel_cmd, states=full_state,u_vector=u_vec)

def amputate_run_data_to_t(sim_run: SimObjectOutput, t_cutoff: float):
    index = np.where(sim_run.time > t_cutoff)[0]
    print(sim_run.time[index])
    print(sim_run.trans_true[index,:])    
    return SimObjectOutput(
        time = sim_run.time[index],
        trans_true=sim_run.trans_true[index,:],
        trans_cmd=sim_run.trans_cmd[index,:],
        trans_vel_true=sim_run.trans_vel_true[index,:],
        trans_vel_cmd=sim_run.trans_vel_cmd[index,:],
        rot_true=sim_run.rot_true[index,:],
        rot_cmd=sim_run.rot_cmd[index,:],
        rot_vel_true=sim_run.rot_vel_true[index,:],
        rot_vel_cmd=sim_run.rot_vel_cmd[index,:],
        states=sim_run.states[index,:],
        u_vector=sim_run.u_vector[index,:],
    )

    
def plot_run(sim_run: SimObjectOutput):
    position_plot(sim_run)
    loss_plot(sim_run)
    reward_plot(sim_run)
    attitude_plot(sim_run)
    velocity_plot(sim_run)
    acceleration_plot(sim_run)
    jerk_plot(sim_run)

def jerk_plot(sim_run: SimObjectOutput, output_dir="data/results/object"):
    acc_vals = sim_run.compute_acceleration()
    meters = mpl.ticker.EngFormatter("m")
    newtons = mpl.ticker.EngFormatter("N")
    seconds = mpl.ticker.EngFormatter("s")
    radians = mpl.ticker.EngFormatter("rad")
    time = acc_vals["jerk_time"]
    obj_x = acc_vals["trans_jerk_true"][:,0]
    obj_y = acc_vals["trans_jerk_true"][:,1]
    obj_z = acc_vals["trans_jerk_true"][:,2]

    fig, accplot = plt.subplots()
    plothusly(accplot, time, obj_z, 
            xtitle="Time [sec]",
            ytitle="Jerk [m/s^3]", 
            datalabel="Z Jerk", 
            title="Sim Obj Jerk")
    plothus(accplot, time, obj_y, datalabel="Y Jerk")
    plothus(accplot, time, obj_x, datalabel="X Jerk")

    accplot.xaxis.set_major_formatter(seconds)
    accplot.yaxis.set_major_formatter(meters)

    fig.savefig(f"{output_dir}_jrk.png")

def acceleration_plot(sim_run: SimObjectOutput, output_dir="data/results/object"):
    acc_vals = sim_run.compute_acceleration()
    mps2 = mpl.ticker.EngFormatter("m/s^2")
    seconds = mpl.ticker.EngFormatter("s")
    time = acc_vals["accel_time"]
    obj_x = acc_vals["trans_accel_true"][:,0]
    obj_y = acc_vals["trans_accel_true"][:,1]
    obj_z = acc_vals["trans_accel_true"][:,2]

    fig, accplot = plt.subplots()
    plothusly(accplot, time, obj_z, 
            xtitle="Time [sec]",
            ytitle="Acceleration [m/s^2]", 
            datalabel="Z Acceleration", 
            title="Sim Obj Acceleration")
    plothus(accplot, time, obj_y, datalabel="Y Acceleration")
    plothus(accplot, time, obj_x, datalabel="X Acceleration")

    accplot.xaxis.set_major_formatter(seconds)
    accplot.yaxis.set_major_formatter(mps2)

    fig.savefig(f"{output_dir}_acc.png")

def position_plot(sim_run: SimObjectOutput, output_dir="data/results/object_position"):

    meters = mpl.ticker.EngFormatter("m")
    newtons = mpl.ticker.EngFormatter("N")
    seconds = mpl.ticker.EngFormatter("s")
    radians = mpl.ticker.EngFormatter("rad")
    time = sim_run.time
    obj_x = sim_run.trans_true[:,0]
    obj_y = sim_run.trans_true[:,1]
    obj_z = sim_run.trans_true[:,2]

    cmd_x, cmd_y, cmd_z = sim_run.trans_cmd[:,0], sim_run.trans_cmd[:,1], sim_run.trans_cmd[:,2],

    fig, zplot = plt.subplots()
    plothusly(zplot, time, obj_z, 
            xtitle="Time [sec]",
            ytitle="Position [m]]", 
            datalabel="Z Position", 
            title="Sim Obj Position")
    plothus(zplot, time, obj_y, datalabel="Y Position")
    plothus(zplot, time, obj_x, datalabel="X Position")

    plothus(zplot, time, cmd_z, datalabel="Z Command", linestyle='--')
    plothus(zplot, time, cmd_x, datalabel="X Command", linestyle='--')
    plothus(zplot, time, cmd_y, datalabel="Y Command", linestyle='--')

    zplot.xaxis.set_major_formatter(seconds)
    zplot.yaxis.set_major_formatter(meters)

    fig.savefig(f"{output_dir}_pos.png")

def velocity_plot(sim_run: SimObjectOutput, output_dir="data/results/object"):

    meters = mpl.ticker.EngFormatter("m")
    newtons = mpl.ticker.EngFormatter("N")
    seconds = mpl.ticker.EngFormatter("s")
    radians = mpl.ticker.EngFormatter("rad")
    mps = mpl.ticker.EngFormatter("m/s")
    time = sim_run.time
    obj_x = sim_run.trans_vel_true[:,0]
    obj_y = sim_run.trans_vel_true[:,1]
    obj_z = sim_run.trans_vel_true[:,2]

    fig, zplot = plt.subplots()
    plothusly(zplot, time, obj_z, 
            xtitle="Time [sec]",
            ytitle="Position [m]]", 
            datalabel="Z Velocity", 
            title="Sim Obj Position")
    plothus(zplot, time, obj_y, datalabel="Y Velocity")
    plothus(zplot, time, obj_x, datalabel="X Velocity")

    zplot.xaxis.set_major_formatter(seconds)
    zplot.yaxis.set_major_formatter(mps)

    fig.savefig(f"{output_dir}_vel.png")

def loss_plot(sim_run: SimObjectOutput, output_dir="data/results/object"):
    seconds = mpl.ticker.EngFormatter("s")
    
    loss = calculate_loss(sim_run)

    fig, zplot = plt.subplots()
    plothusly(zplot, sim_run.time, loss, 
            xtitle="Time [sec]",
            ytitle="Position [m]]", 
            datalabel="Z Position", 
            title="Sim Obj Position")
    # plothus(zplot, time, obj_y, datalabel="Y Position")
    # plothus(zplot, time, obj_x, datalabel="X Position")

    zplot.xaxis.set_major_formatter(seconds)
    # zplot.yaxis.set_major_formatter(meters)

    fig.savefig(f"{output_dir}_loss.png")

def plot_3D_trajectories(sim_run: SimObjectOutput):
    """
    Plot 3D trajectories of all objects using Plotly.

    Args:
    combined_data (list): List of tuples (object_id, dataframe) as returned by combine_csv_files.

    Returns:
    plotly.graph_objects.Figure: The Plotly figure object.
    """
    fig = go.Figure()

    obj_x = sim_run.trans_true[:,0]
    obj_y = sim_run.trans_true[:,1]
    obj_z = sim_run.trans_true[:,2]

    # for object in combined_data:
    # object_id = object.id
    # df = object.data
    fig.add_trace(go.Scatter3d(
        x=obj_x, y=obj_y, z=obj_z,
        mode='lines',
        name="sample"
    ))

    fig.update_layout(
        title="3D Trajectories of Objects",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        legend_title="Objects"
    )

    return fig

def plothusly(ax, x, y, *, xtitle='', ytitle='',
              datalabel='', title='', linestyle='-',
              marker=''):
    """
    A little function to make graphing less of a pain.
    Creates a plot with titles and axis labels.
    Adds a new line to a blank figure and labels it.

    Parameters
    ----------
    ax : The graph object
    x : X axis data
    y : Y axis data
    xtitle : Optional x axis data title. The default is ''.
    ytitle : Optional y axis data title. The default is ''.
    datalabel : Optional label for data. The default is ''.
    title : Graph Title. The default is ''.

    Returns
    -------
    out : Resultant graph.

    """
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_title(title)
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle=linestyle,
                  marker = marker)
    ax.grid(True)
    ax.legend(loc='best')
    return out

def plothus(ax, x, y, *, datalabel='', linestyle = '-',
            marker = ''):
    """
    A little function to make graphing less of a pain

    Adds a new line to a blank figure and labels it
    """
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle = linestyle,
                  marker = marker)
    ax.legend(loc='best')
    return out

def TransformationMatrix(phi, theta, psi, num_blocks=4):
    """
    Transforms 6 DOF state vector from body axis to earth axis

    Parameters
    ----------
    phi : Roll angle in radians
    theta : Pitch angle in radians
    psi : Yaw angle in radians

    Returns
    -------
    array : 12x12 transformation matrix
    """
    R = np.array([[np.cos(psi)*np.cos(theta), np.cos(psi)*np.sin(theta)*np.sin(phi)-np.sin(psi)*np.cos(phi), np.cos(psi)*np.sin(theta)*np.cos(phi)+np.sin(psi)*np.sin(phi)],
                  [np.sin(psi)*np.cos(theta), np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(psi)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi)-np.cos(psi)*np.sin(phi)],
                  [-1*np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)]])

    return np.kron(np.eye(num_blocks, dtype=float), R)

def compute_trajectory_reward(sim_run: SimObjectOutput):
    def reward_func(time, true, cmd):
        loss = 0
        for i in range(1,len(time)):
            loss -= np.linalg.norm(
                np.abs(cmd[i, :] - true[i, :])**2
                - 0.1 * (np.abs(true[i-1, :] - true[i, :])**2
                        + np.abs(cmd[i, :] - true[i, :])**2)
            )
        return loss
        
    # RMS of Position
    pos_error = reward_func(sim_run.time, sim_run.trans_true, sim_run.trans_cmd)
    

    vel_error = reward_func(sim_run.time, sim_run.trans_vel_true, sim_run.trans_vel_cmd)

    rot_error = reward_func(sim_run.time, sim_run.rot_true, sim_run.rot_cmd)

    rot_vel_error = reward_func(sim_run.time, sim_run.rot_vel_true, sim_run.rot_vel_cmd)

    # print(pos_error)
    return 10 * pos_error + 1 * vel_error + 100 * rot_error + 1 * rot_vel_error

# Rewrite later
def keplerian_to_cartesian(a, e, i, Ω, ω, ν, μ=398600.4418):
    """
    Convert Keplerian elements to Cartesian state vector.
    
    Parameters:
    a : float
        Semi-major axis (km)
    e : float
        Eccentricity
    i : float
        Inclination (radians)
    Ω : float
        Right ascension of the ascending node (radians)
    ω : float
        Argument of periapsis (radians)
    ν : float
        True anomaly (radians)
    μ : float, optional
        Gravitational parameter (km^3/s^2), defaults to Earth's

    Returns:
    r : numpy.ndarray
        Position vector in ECI frame (km)
    v : numpy.ndarray
        Velocity vector in ECI frame (km/s)
    """
    
    # Calculate position and velocity in orbital plane
    r_mag = a * (1 - e**2) / (1 + e * np.cos(ν))
    
    r_p = r_mag * np.array([np.cos(ν), np.sin(ν), 0])
    v_p = np.sqrt(μ / (a * (1 - e**2))) * np.array([-np.sin(ν), e + np.cos(ν), 0])
    
    # Rotation matrices
    R_ω = np.array([
        [np.cos(ω), -np.sin(ω), 0],
        [np.sin(ω), np.cos(ω), 0],
        [0, 0, 1]
    ])
    
    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    
    R_Ω = np.array([
        [np.cos(Ω), -np.sin(Ω), 0],
        [np.sin(Ω), np.cos(Ω), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = R_Ω @ R_i @ R_ω
    
    # Transform to ECI frame
    r = R @ r_p 
    v = R @ v_p
    
    return r, v

# BASIC SCENARIO    
def test_scenario():
    # Earth Environment
    environments = [
        PointMassGravity(mass=5.97219E24),
    ]
    r_iss, v_iss = keplerian_to_cartesian(a=414+6.378E3, e=0.0009143, i=np.deg2rad(51.6367), Ω=np.deg2rad(114.1365), ω=np.deg2rad(50.0139), ν=np.deg2rad(310.1651),μ=398600.4418)
    r_iss*=1E3
    v_iss*=1E3
    # print(r_iss)
    # print(np.sqrt(r_iss[0]**2+r_iss[1]**2+r_iss[2]**2))
    # print(400.0E3+6.378E6)
    init_state=np.array([
        [r_iss[0]],    
        [r_iss[1]],
        [r_iss[2]],
        [v_iss[0]],     
        [v_iss[1]],
        [v_iss[2]],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        ])
    mass_sim_ISS = SimObject(
        name="ISS", 
        mass=1.0, 
        # graphical_path="data/test_object/default_cube.obj", 
        physics_type="RK4",
        init_state= init_state,
        graphical_elements=[GraphicalElement("CubeModel", "data/test_object/default_cube.obj", [0,0,0], [0,0,0], [0,0,0], [1,0,0,1], [100E3*1E-6,100E3*1E-6,100E3*1E-6])],
        recording_sample_time=0.0,
        max_recorder_buffer_steps=int(99E10)
    )

    earth_graphical = SimObject(
        name="earth",
        mass=0,
        # graphical_path = "data/test_object/default_sphere.obj",
        physics_type="Static",
        graphical_elements=[GraphicalElement("EarthModel", "data/test_object/default_sphere.obj", [0,0,0], [0,0,0], [0,0,0], [0,0,1,1], [6.378E6*1E-6,6.378E6*1E-6,6.378E6*1E-6])]
    )


    scene = Scenario(objects=[earth_graphical, mass_sim_ISS], environments=environments, min_dt=1E-3, end_time=2*60*90)
    return scene

def test_constellation_scenario():

    # Earth Environment
    environments = [
        PointMassGravity(mass=5.97219E24),
    ]

    run_name = "test_constellation"
    sample_time = 60
    max_steps = 1024

    objects = []
    longitudes = np.linspace(0, 360, 10)

    earth_graphical = SimObject(
        name="earth",
        mass=0,
        # graphical_path = "data/test_object/default_sphere.obj",
        physics_type="Static",
        graphical_elements=[GraphicalElement("EarthModel", "data/test_object/default_sphere.obj", [0,0,0], [0,0,0], [0,0,0], [0,0,1,1], [6.378E6*1E-6,6.378E6*1E-6,6.378E6*1E-6])],
        run_name=run_name,
        recording_sample_time=sample_time,
        max_recorder_buffer_steps=max_steps
    )
    objects.append(earth_graphical)
    for i, longitude in enumerate(longitudes):

        r_iss, v_iss = keplerian_to_cartesian(a=414+6.378E3, e=0.0009143, i=np.deg2rad(51.6367), Ω=np.deg2rad(longitude), ω=np.deg2rad(50.0139), ν=np.deg2rad(310.1651),μ=398600.4418)
        r_iss*=1E3
        v_iss*=1E3
        # print(r_iss)
        # print(np.sqrt(r_iss[0]**2+r_iss[1]**2+r_iss[2]**2))
        # print(400.0E3+6.378E6)
        init_state=np.array([
            [r_iss[0]],    
            [r_iss[1]],
            [r_iss[2]],
            [v_iss[0]],     
            [v_iss[1]],
            [v_iss[2]],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            ])
        mass_sim_ISS = SimObject(
            name=f"ISS_{i}", 
            mass=1.0, 
            # graphical_path="data/test_object/default_cube.obj", 
            physics_type="RK4",
            init_state= init_state,
            graphical_elements=[GraphicalElement("CubeModel", "data/test_object/default_cube.obj", [0,0,0], [0,0,0], [0,0,0], [1,0,0,1], [100E3*1E-6,100E3*1E-6,100E3*1E-6])],
            run_name=run_name,
            recording_sample_time=sample_time,
            max_recorder_buffer_steps=max_steps
        )
        objects.append(mass_sim_ISS)

    


    scene = Scenario(scenario_name=run_name, objects=objects, environments=environments, min_dt=1E-3, end_time=7*60*60*24)
    scene.create_run_json()

def test_quadcopter():
    environments = [
        # ConstantField(),
        PointMassGravity(mass=5.97219E24, position=np.array([0.0, 0.0, -6.378E6]))
    ]

    run_name = "test_drone"
    sample_time = 0.1
    max_steps=100000
    objects=[]

    init_state = np.zeros((12,1))

    I = np.array([
        [0.0025, 0.0, 0.0],
        [0.0, 0.0025, 0.0],
        [0.0, 0.0, 0.005]
    ])

    K = np.array([
        [0.0],  # x
        [0.0],  # y
        [20.0],  # z
        [0.0],  # x'
        [0.0],  # 
        [1.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
    ])

    B = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    fc = FlightComputer(sample_time=0.0, sensors=None, K=K)
    components = [
        IdealThruster(time_constant=0.01),
        # IdealThruster(time_constant=0.01)
    ]

    drone = SimObject(
        name="TestDrone",
        mass=0.25,
        physics_type="RK4",
        I=I,
        dependent_components=components,
        B=B,
        graphical_elements=[
            GraphicalElement("DroneModel", "data/test_object/default_sphere.obj", [0,0,0], [0,0,0], [0,0,0], [1,0,0,1])
        ],
        fc=fc,
        init_state=init_state,
        recording_sample_time=sample_time,
        max_recorder_buffer_steps=max_steps,
        run_name=run_name
        )
    
    objects.append(drone)

    static_cube = SimObject(
        name="StaticCube",
        mass=0.25,
        physics_type="Static",
        I=I,
        graphical_elements=[
            GraphicalElement("DroneModel", "data/test_object/default_cube.obj", [0,0,0], [0,0,0], [0,0,0], [0.5,0.5,0.5,1])
        ],
        init_state=init_state,
        run_name=run_name
        )
    
    objects.append(static_cube)

    terrain = SimObject(
        name="HighPolyTerrain",
        mass=0,
        physics_type="Static",
        I=I,
        graphical_elements=[
            GraphicalElement("TerrainMap", "data/test_object/MASADA_HIGHPOLY.obj", [0,0,0], [0,0,0], [0,0,0], [0.0,1.0,0.0,1.0], [1E3, 1E3, 1E3])],
        init_state=init_state,
        run_name=run_name
    )
    objects.append(terrain)


    scene = Scenario(scenario_name=run_name, objects=objects, environments=environments, min_dt=1E-3, end_time=10.0)
    return scene

def blizzard_test_object_setup(fc=None):

    blizzard_graphical = GraphicalElement(
        name="blizzard_body_model",
        filepath="data/test_object/blizzard.obj",
        relative_position=[0,0,0],
        orientation=[0,0,0],
        rotation=[0,0,0],
        color=[1, 1, 1, 1],
        scale=[1,1,1]
    )

    rotor_positions = [
        [-0.72, -2.982, 1.041+0.15],    # FLT
        [-0.72, 2.982, 1.041+0.15],     # FRT
        [4.220, -2.982, 1.041+0.15],    # BLT
        [4.220, 2.982, 1.041+0.15],     # BRT
        [-0.72, -2.982, 1.041-0.15],    # FLB
        [-0.72, 2.982, 1.041-0.15],     # FRB
        [4.220, -2.982, 1.041-0.15],    # BLB
        [4.220, 2.982, 1.041-0.15],     # BRB
    ]

    components = []
    for position in rotor_positions:
        components.append(
            IdealThruster(time_constant=0.8,
                          position=np.array([position])
                          )
        )
    mass = 2200.0

    mixer = np.array([[0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0, 0],  # X Forces
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Y Forces
                  [1, 1, 1, 1, 1, 1, 1, 1],  # Z Forces
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty 
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty 
                  [3, -3, 3, -3, 3, -3, 3, -3],  # X Moments (Roll)
                  [2.5, 2.5, -2.5, -2.5, 2.5, 2.5, -2.5, -2.5],  # Y Moments (Pitch)
                  [-1, 1, 1, -1, 1, -1, -1, 1]], dtype=float)  # Z Moments (Yaw)
    
    I = np.array([[600, 0, 0],
              [0, 800, 0],
              [0, 0, 800]])
    
    B = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # x'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # y'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # z'
                           [0, 0, 0, 1/mass, 0, 0, 0, 0, 0, 0, 0, 0],       # x''
                           [0, 0, 0, 0, 1/mass, 0, 0, 0, 0, 0, 0, 0],       # y''
                           [0, 0, 0, 0, 0, 1/mass, 0, 0, 0, 0, 0, 0],       # z''
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # phi'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # theta'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # psi'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # phi''
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # theta''
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],           # psi''
                          dtype=float)
    
    B[9:12, 9:12] = np.linalg.inv(I)
    B = B @ mixer
    # print(B)

    K = np.loadtxt("data/todo/default_name/objects/blizzard/Klqr.csv", dtype=float, delimiter=',')

    K[9,1] = -0.1*K[9,6]
    K[9,4] = -0.3*K[9,9]

    K[10,0] = 0.1*K[10,7]
    K[10,3] = 0.3*K[10,10]
    
    K = mixer.transpose() @ -K

    # print(K)

    if fc is None:
        fc=FlightComputer(
                sample_time=0.001,
                sensors=[],
                K=K,
            )


    blizzard = SimObject(
        name="blizzard",
        mass=mass,
        I = I,
        graphical_elements=[blizzard_graphical],
        physics_type="RK4",
        init_state=np.zeros((12,1)),
        dependent_components=components,
        # A = np.array([
        #             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   # x
        #             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   # y
        #             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   # z
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # x'
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # y'
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # z'
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],   # phi
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   # theta
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],   # psi
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # phi'
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # theta'
        #             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float),  # psi',
        B=B,
        # fc=FlightComputer(
        #     sample_time=0.001,
        #     sensors=[],
        #     K=K,
        #     # K = np.zeros((12,4))
        # ),
        fc=fc,
        recording_sample_time=0.001,
        max_recorder_buffer_steps=1000,
        run_name="nn_train_att_1"
    )

    return blizzard

def blizzard_hover_scenario_setup(name="blizzard_hover_test"): 
    
    environments = [
        ConstantField(),
        # PointMassGravity(mass=5.97219E24, position=np.array([0.0, 0.0, -6.378E6]))
    ]

    init_state = np.zeros((12,1))
    init_state[2]=-10

    blizzard_instance = Object_Instance(
        "blizzard",
        init_state,
        "RK4",
        8,
    )

    # object_list = [blizzard_instance for i in range(RUN_NUMS)]
    # print(object_list)
    object_list = [blizzard_instance]

    return Scenario(scenario_name=name, end_time=15.0, min_dt=0.001, objects=object_list, environments=environments)

def blizzard_attitude_stabilization_setup(name="attitude_test"): 
    
    environments = [
        ConstantField(),
        # PointMassGravity(mass=5.97219E24, position=np.array([0.0, 0.0, -6.378E6]))
    ]

    init_state = np.zeros((12,1))
    init_state[3:6] = np.random.rand(3,1) * math.pi
    

    blizzard_instance = Object_Instance(
        "blizzard",
        init_state,
        "RK4",
        8,
    )

    object_list = [blizzard_instance]

    return Scenario(scenario_name=name, end_time=60.0, min_dt=0.001, objects=object_list, environments=environments)

def blizzard_return_to_zero(name="displacement_test"): 
    
    environments = [
        ConstantField(),
        # PointMassGravity(mass=5.97219E24, position=np.array([0.0, 0.0, -6.378E6]))
    ]

    init_state = np.zeros((12,1))
    init_state[0:3] = np.random.rand(3,1) * 10
    

    blizzard_instance = Object_Instance(
        "blizzard",
        init_state,
        "RK4",
        8,
    )

    # object_list = [blizzard_instance for i in range(RUN_NUMS)]
    # print(object_list)
    object_list = [blizzard_instance]

    return Scenario(scenario_name=name, end_time=60.0, min_dt=0.001, objects=object_list, environments=environments)

def nn_train_att_1():
    repeat_job = Job()

    repeat_job.add_object(blizzard_test_object_setup())
    repeat_job.objects[0].fc.set_NN_filepath(repeat_job.path+"/objects/blizzard/blizzard.onnx")

    repeat_job.add_scenario(blizzard_hover_scenario_setup())
    try:
        os.rmdir(repeat_job.path+"/output")
    except:
        pass
    try:
        os.rmdir(repeat_job.path+"/scenarios")
    except:
        pass

    # repeat_job.export_job()

    return repeat_job

def load_all_simulation_runs(output_path):
    """
    Function that loads all sim runs into individual data structs
    """
    data_list = []
    scenarios = os.listdir(output_path)
    # print(f"Scenarios: {scenarios}")
    for scenario in scenarios:
        csv_files = os.listdir(output_path+"/"+scenario)
        # print(f"Output path: {output_path}")
        # print(scenario)
        # print(csv_files)
        [data_list.append(read_from_csv(output_path+"/"+scenario+"/"+file)) for file in csv_files]

    return data_list

def basic_job():
    job = Job()

    fc=FlightComputer(
            sample_time=0.001,
            sensors=[],
            # K=K,
            K = np.zeros((12,8))
        )
    job.add_object(blizzard_test_object_setup(
        # fc
    ))

    mixer = np.array([[0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0, 0],  # X Forces
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Y Forces
                  [1, 1, 1, 1, 1, 1, 1, 1],  # Z Forces
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty 
                  [0, 0, 0, 0, 0, 0, 0, 0],  # Empty 
                  [3, -3, 3, -3, 3, -3, 3, -3],  # X Moments (Roll)
                  [2.5, 2.5, -2.5, -2.5, 2.5, 2.5, -2.5, -2.5],  # Y Moments (Pitch)
                  [-1, 1, 1, -1, 1, -1, -1, 1]], dtype=float)  # Z Moments (Yaw)

    K = np.loadtxt("data/todo/default_name/objects/blizzard/Klqr.csv", dtype=float, delimiter=',')

    K[9,1] = -0.1*K[9,6]
    K[9,4] = -0.3*K[9,9]

    K[10,0] = 0.1*K[10,7]
    K[10,3] = 0.3*K[10,10]
    
    K = mixer.transpose() @ -K

    job.add_scenario(blizzard_hover_scenario_setup())
    [job.add_scenario(blizzard_attitude_stabilization_setup(f"attitude_test_{i}")) for i in range(RUN_NUMS)]
    [job.add_scenario(blizzard_return_to_zero(f"displacemnet_test_{i}")) for i in range(RUN_NUMS)]

    return job

def call_sim():
    # print(os.name)
    if os.name == 'posix':
        os.system('./target/release/sixdof')
    else: 
        os.system('target\\release\\sixdof.exe')
    return load_all_simulation_runs("data/todo/default_name/output")

def calculate_loss(sim_run: SimObjectOutput):
    loss = np.zeros_like(sim_run.time)
    loss += 10*calculate_position_step_loss(sim_run)
    return loss

def calculate_reward_vector(sim_run: SimObjectOutput):
    reward = np.zeros_like(sim_run.time)
    reward += calculate_relevant_velocity_reward(sim_run)
    reward += calculate_relevant_acceleration_reward(sim_run)
    reward += calculate_relevant_jerk_reward(sim_run)
    reward += calculate_position_reward(sim_run)
    reward += calculate_rotation_reward(sim_run)
    reward += calculate_relevant_rotation_velocity_reward(sim_run)

    return reward

def reward_plot(sim_run: SimObjectOutput, output_dir='data/results/object'):
    reward = calculate_reward_vector(sim_run)
    seconds = mpl.ticker.EngFormatter("s")
    fig, zplot = plt.subplots()
    plothusly(zplot, sim_run.time, reward, 
            xtitle="Time [sec]",
            ytitle="Position [m]]", 
            datalabel="Z Position", 
            title="Sim Obj Reward")

    zplot.xaxis.set_major_formatter(seconds)
    fig.savefig(f"{output_dir}_reward.png")

def attitude_plot(sim_run: SimObjectOutput, output_dir='data/results/object'):
    fig, zplot = plt.subplots()
    seconds = mpl.ticker.EngFormatter("s")
    time = sim_run.time
    pitch = sim_run.rot_true[:,0]
    roll = sim_run.rot_true[:,1]
    yaw = sim_run.rot_true[:,2]

    fig, zplot = plt.subplots()
    plothusly(zplot, time, pitch, 
            xtitle="Time [sec]",
            ytitle="Position [m]]", 
            datalabel="Pitch", 
            title="Sim Obj Attitude")
    plothus(zplot, time, roll, datalabel="Roll")
    plothus(zplot, time, yaw, datalabel="Yaw")
    fig.savefig(f"{output_dir}_attitude.png")

def calculate_position_reward(sim_run: SimObjectOutput):
    err = sim_run.trans_cmd - sim_run.trans_true
    reward = np.array(
        [1E4*np.exp(-np.linalg.norm(err_point)) for err_point in(err)]
    )
    return reward

def calculate_rotation_reward(sim_run: SimObjectOutput):
    err = sim_run.rot_cmd - sim_run.rot_true
    reward = np.array(
        [1E4*np.exp(-np.linalg.norm(err_point)) for err_point in(err)]
    )
    return reward
    
def calculate_relevant_velocity_reward(sim_run: SimObjectOutput):
    reward = np.zeros_like(sim_run.time)
    dt = np.abs(sim_run.time[1]-sim_run.time[0])
    err = sim_run.trans_cmd - sim_run.trans_true
    reward = np.array(
        [np.linalg.norm(err_point)*np.dot(vel_true, err_point) 
         + 100*np.exp(-np.linalg.norm(vel_true)*np.exp(-np.linalg.norm(err_point))) for vel_true, err_point in zip(sim_run.trans_vel_true, err)]
        )
    reward = np.max([np.zeros_like(reward), reward], axis=0)
    reward = np.min([100*np.ones_like(reward), reward], axis=0)
    return reward

def calculate_relevant_acceleration_reward(sim_run: SimObjectOutput):
    calculated_info = sim_run.compute_acceleration()

    accel_vec = np.concatenate([
        np.zeros((1,3)), calculated_info["trans_accel_true"]
    ])
    err = sim_run.trans_cmd - sim_run.trans_true
    reward = np.array(
        [np.linalg.norm(err_point)*np.dot(acc_true, err_point)
         + 100*np.exp(-np.linalg.norm(acc_true)*np.exp(-10*np.linalg.norm(err_point))) for acc_true, err_point in zip(accel_vec, err)]
        )   
    reward = np.max([np.zeros_like(reward), reward], axis=0)
    reward = np.min([10*np.ones_like(reward), reward], axis=0)
    return reward

def calculate_relevant_jerk_reward(sim_run: SimObjectOutput):
    calculated_info = sim_run.compute_acceleration()

    jerk_vec = np.concatenate([
        np.zeros((3,3)), calculated_info["trans_jerk_true"]
    ])
    # print(calculated_info["trans_jerk_true"])
    err = sim_run.trans_cmd - sim_run.trans_true
    reward = np.array(
        [np.linalg.norm(err_point)*np.dot(jerk_true, err_point)
         + 0.001*np.exp(-np.linalg.norm(jerk_true)*np.exp(-1*np.linalg.norm(err_point))) for jerk_true, err_point in zip(jerk_vec, err)]
        )   
    reward = np.max([np.zeros_like(reward), reward], axis=0)
    reward = np.min([10*np.ones_like(reward), reward], axis=0)
    return reward

def calculate_relevant_rotation_velocity_reward(sim_run: SimObjectOutput):
    reward = np.zeros_like(sim_run.time)
    dt = np.abs(sim_run.time[1]-sim_run.time[0])
    err = sim_run.rot_cmd - sim_run.rot_true
    reward = np.array(
        [np.linalg.norm(err_point)*np.dot(vel_true, err_point) for vel_true, err_point in zip(sim_run.rot_vel_true, err)]
        )
    
    return reward

def calculate_position_step_loss(sim_run: SimObjectOutput):
    err = sim_run.trans_cmd - sim_run.trans_true
    pos_err_log_loss = np.zeros_like(sim_run.time)
    for i, err_point in enumerate(err):
        pos_err_log_loss[i] = np.linalg.norm(err_point)
    return pos_err_log_loss

def nn_att_2(fc=None):
    job = Job()
    if fc is None:
        job.add_object(blizzard_test_object_setup(
            fc=FlightComputer(sample_time=0.001, sensors=[], K = BlizzardController())
        ))
    else:
        job.add_object(blizzard_test_object_setup(
            fc=fc
        ))
    job.add_scenario(blizzard_hover_scenario_setup())
    job.objects[0].fc.set_NN_filepath(job.path+"/objects/blizzard/blizzard.onnx")
    [job.add_scenario(blizzard_attitude_stabilization_setup(f"attitude_test_{i}")) for i in range(RUN_NUMS)]
    [job.add_scenario(blizzard_return_to_zero(f"displacemnet_test_{i}")) for i in range(RUN_NUMS)]
    return job

def calculate_reward_from_multiple_outputs(sim_run_vector):
    reward_vec = None
    for run in sim_run_vector:
        if reward_vec is None:
            reward_vec = calculate_reward_vector(run)
        else:
            reward_vec = np.concatenate([reward_vec, calculate_reward_vector(run)], axis=0)

    return reward_vec

def add_gradient_noise(model, noise_factor=0.01):
    for param in model.parameters():
        if param.grad is not None:
            param.grad += noise_factor * torch.randn_like(param.grad)

def optimize_model(job: Job, sim_data_vector: [SimObjectOutput], model: nn.Module, optimizer: torch.optim.Optimizer, 
                  *, num_epochs: int = 50, batch_size: int = 64, gamma: float = 0.99, pretraining_epochs=100,
                  plateau_patience: int=5, noise_factor: float = 0.01, loss_threshold: float = 10e3):
    """
    Optimize the neural network model using collected simulation data.
    
    Args:
        job: Simulation job object
        sim_run: Simulation output containing states and actions
        model: Neural network model to optimize
        optimizer: PyTorch optimizer
        num_epochs: Number of training epochs
        batch_size: Size of training batches
        gamma: Discount factor for future rewards
    """
    # Convert simulation data to tensors
    states, next_states, actions = get_data_from_sim_run_list(sim_data_vector)
    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(calculate_reward_from_multiple_outputs(sim_data_vector), dtype=torch.float32)
    
    # Setup learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                patience=plateau_patience)
    
    # Setup plateau detection
    best_loss = float('inf')
    plateau_count = 0
    plateau_threshold = 1e-4
    
    # Pretraining epochs
    for epoch in range(pretraining_epochs):
        n_samples = len(states)
        n_batches = n_samples // batch_size
        total_loss = 0.0
        
        # Shuffle data for each epoch
        indices = torch.randperm(n_samples)
        
        for batch in range(n_batches):
            optimizer.zero_grad()
            
            # Get batch indices
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_states = states[batch_indices]
            batch_next_states = next_states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_rewards = rewards[batch_indices]
            
            # Compute current Q-values and get the Q-values for the taken actions
            current_q_values = model(batch_states)
            action_q_values = torch.sum(current_q_values * batch_actions, dim=1)
            
            # Compute next Q-values with target network
            with torch.no_grad():
                next_q_values = model(batch_next_states)
                max_next_q_values, _ = next_q_values.max(dim=1)
                target_q_values = batch_rewards + gamma * max_next_q_values
            
            # Compute loss - ensure dimensions match
            loss = nn.MSELoss()(action_q_values, target_q_values)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        # Print epoch statistics
        avg_loss = total_loss / n_batches

        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Check for plateau
        if abs(avg_loss - best_loss) < plateau_threshold:
            plateau_count += 1
            if plateau_count >= 3:  # If stuck for 3 epochs
                print(f"Plateau detected! Increasing noise factor temporarily...")
                temp_noise = noise_factor * 2.0  # Temporarily increase noise
                add_gradient_noise(model, temp_noise)
                plateau_count = 0
        else:
            plateau_count = 0
            if avg_loss < best_loss:
                best_loss = avg_loss

        print(f"Pretraining epoch {epoch + 1}/{num_epochs}: Average Loss = {avg_loss:,.4f}")

    # Calculate number of batches
    for epoch in range(num_epochs):
        n_samples = len(states)
        n_batches = n_samples // batch_size
        total_loss = 0.0
        
        # Shuffle data for each epoch
        indices = torch.randperm(n_samples)
        
        for batch in range(n_batches):
            optimizer.zero_grad()
            
            # Get batch indices
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_states = states[batch_indices]
            batch_next_states = next_states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_rewards = rewards[batch_indices]
            
            # Compute current Q-values and get the Q-values for the taken actions
            current_q_values = model(batch_states)
            action_q_values = torch.sum(current_q_values * batch_actions, dim=1)
            
            # Compute next Q-values with target network
            with torch.no_grad():
                next_q_values = model(batch_next_states)
                max_next_q_values, _ = next_q_values.max(dim=1)
                target_q_values = batch_rewards + gamma * max_next_q_values
            
            # Compute loss - ensure dimensions match
            loss = nn.MSELoss()(action_q_values, target_q_values)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Export model and run simulation
        if epoch % 10 == 0:  # Run sim every 10 batches to save time
            job.export_job()
            sim_data_vector = call_sim()
            sim_run=sim_data_vector[0]

            states, next_states, actions = get_data_from_sim_run_list(sim_data_vector)
            
            states = torch.tensor(states, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(calculate_reward_from_multiple_outputs(sim_data_vector), dtype=torch.float32)
            
            # Get new simulation results and update rewards if needed
            # Note: You might want to implement a method to get new sim results
                
        # Print epoch statistics
        avg_loss = total_loss / n_batches
        # Update learning rate scheduler
        scheduler.step(avg_loss)
        
        # Check for plateau
        if abs(avg_loss - best_loss) < plateau_threshold:
            plateau_count += 1
            if plateau_count >= 3:  # If stuck for 3 epochs
                print(f"Plateau detected! Increasing noise factor temporarily...")
                temp_noise = noise_factor * 2.0  # Temporarily increase noise
                add_gradient_noise(model, temp_noise)
                plateau_count = 0
        else:
            plateau_count = 0
            if avg_loss < best_loss:
                best_loss = avg_loss
        print(f"Epoch {epoch + 1}/{num_epochs}: Average Loss = {avg_loss:,.4f}")
        
        # Optional: Early stopping if loss is small enough
        if avg_loss < 1e-4:
            print("Converged! Stopping early.")
            break
        position_plot(sim_run)
        loss_plot(sim_run)
        reward_plot(sim_run)
        attitude_plot(sim_run)
    
    return model

def get_data_from_sim_run_list(sim_runs: [SimObjectOutput], t_cutoff=0.0):
    new_sim_list = [amputate_run_data_to_t(sim_run, t_cutoff) for sim_run in sim_runs]
    states, next_states, actions = None,None,None
    for run in sim_runs:
        if states is None:
            states, next_states, actions = run.to_vector()
        else:
            state_local, next_state_local, actions_local = run.to_vector()
            states = np.concatenate([states, next_state_local], axis=0)
            next_states = np.concatenate([next_states, state_local], axis=0)
            actions = np.concatenate([actions, actions_local], axis=0)
    # print(states.shape)
    return states, next_states, actions

def timing_guidance_computer_test():
    times = [0, 15, 30]
    step = np.zeros((12,1))
    step[2]=10.0
    states = [
        np.zeros((12,1)),
        step,
        np.zeros((12,1))
    ]
    guidance = TimingGuidanceComputer(times, states)
    job = Job()
    job.add_object(
        blizzard_test_object_setup()
    )
    job.objects[0].fc.add_guidance_system(guidance)
    job.add_scenario(blizzard_hover_scenario_setup())
    job.scenarios[0].end_time=60
    return job

def waypoint_guidance_computer_test():
    points = [
        np.zeros((3,1)),
        np.ones((3,1)),
        np.zeros((3,1)),
        -np.ones((3,1)),
        -10*np.ones((3,1)),
        np.zeros((3,1)),
        10*np.ones((3,1)),
    ]
    guidance = WaypointGuidanceComputer(points, tolerance=0.1, hold_time=2)
    job = Job()
    job.add_object(
        blizzard_test_object_setup()
    )
    job.objects[0].fc.add_guidance_system(guidance)
    job.add_scenario(blizzard_hover_scenario_setup())
    job.scenarios[0].end_time=60
    return job

def step_response_timing_test(*, fc=None, t_step=5, mag_step=10):
    times = [0, t_step, ]
    step = np.zeros((12,1))
    step[2]=mag_step
    states = [
        np.zeros((12,1)),
        step,
    ]
    guidance = TimingGuidanceComputer(times, states)
    job = Job()
    job.add_object(
        blizzard_test_object_setup(fc=fc)
    )
    job.objects[0].fc.add_guidance_system(guidance)
    job.add_scenario(blizzard_hover_scenario_setup())
    job.scenarios[0].end_time=60
    return job


def step_response_scenario(name="step_response_test", t_step=10, mag_step=5):
    environments = [
        ConstantField(),
        # PointMassGravity(mass=5.97219E24, position=np.array([0.0, 0.0, -6.378E6]))
    ]

    init_state = np.zeros((12,1))
    init_state[3:6] = np.random.rand(3,1) * math.pi
    

    blizzard_instance = Object_Instance(
        "blizzard",
        init_state,
        "RK4",
        8,
    )
    
    object_list = [blizzard_instance]

    return Scenario(scenario_name=name, end_time=60.0, min_dt=0.001, objects=object_list, environments=environments)

if __name__ == "__main__":
    primer_job = step_response_timing_test(t_step=25)
    primer_job.export_job()
    # print(primer_job.path)
    sim_data_vector = call_sim()
    
    sim_run = read_from_csv("/home/jack/Documents/Projects/sixdof/data/todo/default_name/output/blizzard_hover_test/object_0_blizzard.csv")
    sim_run = amputate_run_data_to_t(sim_run, 10)
    plot_run(sim_run)
    # job = nn_att_2()

    # # sim_data = read_from_csv("data/todo/default_name/output/blizzard_hover_test/object_0_blizzard.csv")
    # # print(job.objects[0].fc.K.parameters())
    # # print(optimizer)
    # # print(sim_data.trans_true[:,2])
    # # plot_3D_trajectories(sim_data).show()
    
    # position_plot(sim_data)
    # loss_plot(sim_data)
    # reward_plot(sim_data)
    # optimizer = torch.optim.Adam(job.objects[0].fc.K.parameters(), lr=LR)
    optimize_model(job, sim_data_vector, job.objects[0].fc.K, optimizer, num_epochs=100, batch_size=1256, pretraining_epochs=300)