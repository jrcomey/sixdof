import json
import numpy as np
import pretty_errors
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import random
import math
import onnx
# torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.double)
import pandas as pd
from analyzedata import *
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

    def __init__(self, sample_time, sensors, K):

        self.sample_time = sample_time
        self.K = K

        if isinstance(self.K, np.ndarray):
            self.fcType = "StateFeedback"
        elif isinstance(self.K, nn.Module):
            self.fcType = "NeuralNet"

    def set_NN_filepath(self, filepath):
        self.NN_filepath = filepath

    def create_json_dict(self):
        if self.fcType=="NeuralNet":

            dummy_input=torch.zeros(1,12).to(torch.float32)
            model = self.K.to(torch.float32)

            # for param in model.parameters():
            #     param.data = param.data.to(torch.float32)

            # print("Model dtype:", next(model.parameters()).dtype)
            # print("Input dtype:", dummy_input.dtype)
            exported_program: torch.export.ExportedProgram =  torch.onnx.export(
            model,
            dummy_input,
            self.NN_filepath,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            # input_types=[torch.float32],
            # output_types=[torch.float32],
            # Explicitly specify float64
            # custom_opsets={'': 17}
            )
            # print(exported_program)

            # onnx_model = onnx.load(self.NN_filepath)
            # for tensor in onnx_model.graph.initializer:
            #     print(f"Tensor {tensor.name} dtype: {onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)}")
            # else:
            #     pass

        if self.fcType=="StateFeedback":
            data = {
            "fcType": self.fcType,
            "sample_time": self.sample_time,
            "K": sum(self.K.tolist(), [])
        }
        elif self.fcType=="NeuralNet":
            data = {
                "fcType": self.fcType,
                "sample_time": self.sample_time,
                "NN_filepath": self.NN_filepath
            }
        return data
    
    def to_json_str(self):
        return json.dumps(self.create_json_dict())

        
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
        data= {
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

# @dataclass
# class SimJobOutput:

# @dataclass
# class SimScenarioOutput:

@dataclass
class SimObjectOutput:
    time:           np.ndarray      # Time vector
    trans_true:     np.ndarray      # True translational position
    trans_cmd:      np.ndarray      # Commanded translational position
    trans_vel_true: np.ndarray      # True translational velocity
    trans_vel_cmd:  np.ndarray      # Commanded translational velocity
    rot_true:       np.ndarray      # True rotational position
    rot_cmd:        np.ndarray      # True rotational command
    rot_vel_true:   np.ndarray
    rot_vel_cmd:    np.ndarray
    # trans_est:      np.ndarray      # Estimated State
    # inputs:         np.ndarray      # inputs

    def read_from_csv(filepath):
        df = pd.read_csv(filepath)
        time = df["Time"].to_numpy()
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
        
        return SimObjectOutput(time, state_true, state_cmd,trans_vel_true=trans_vel_true, trans_vel_cmd=trans_vel_cmd, rot_true=rot_true, rot_cmd=rot_cmd, rot_vel_true=rot_vel_true, rot_vel_cmd=rot_vel_cmd)

class BlizzardController(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 12, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(12, 8, dtype=torch.float32)
        )
        # print(self.network)

    def forward(self, x):
        return self.network(x)

# class BlizzardController(nn.Module):
#     def __init__(self, input_dim=12, output_dim=8):
#         super(BlizzardController, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 144)
#         self.relu = nn.ReLU(),
#         nn.Linear(144, 144, dtype=torch.float32),
#         nn.ReLU(),
#         nn.Linear(144, 144, dtype=torch.float32),
#         nn.ReLU(),
#         nn.Linear(144, 144, dtype=torch.float32),
#         nn.ReLU(),
#         nn.Linear(144, 144, dtype=torch.float32),
#         nn.ReLU(),
#         nn.Linear(144, 144, dtype=torch.float32),
#         nn.ReLU(),
#         nn.Linear(144, 144, dtype=torch.float32),
#         nn.ReLU(),
#         self.fc2 = nn.Linear(128, output_dim)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x  # Output is a control vector U

    
def compute_trajectory_loss(sim_run: SimObjectOutput):

    # RMS of Position
    pos_error = np.mean(
        np.sqrt(
            np.sum(
                (sim_run.trans_cmd-sim_run.trans_true)**2
            )
        )
    )

    

    vel_error = np.mean(
        np.sqrt(
            np.sum(
                (sim_run.trans_vel_cmd-sim_run.trans_vel_true)**2
            )
        )
    )

    rot_error = np.mean(
        np.sqrt(
            np.sum(
                (sim_run.rot_cmd-sim_run.rot_true)**2
            )
        )
    )

    rot_vel_error = np.mean(
        np.sqrt(
            np.sum(
                (sim_run.rot_vel_cmd-sim_run.rot_vel_true)**2
            )
        )
    )

    return 5 * pos_error + 1 * vel_error + 5 * rot_error + 1 * rot_vel_error

# def train_step(model, optimizer, batch_of_runs):
#     optimizer.zero_grad()

#     total_loss = 0
#     for run in batch_of_runs:
#         pass

# def train_model(n_epochs, batch_size=32):
#     model = BlizzardController()
#     optimizer = torch.optim.Adam(model.parameters())

#     for epoch in range(n_epochs):
#         # create and run job here
#         loss = train_step(model, optimizer, batch_runs)

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

def blizzard_test_object_setup():

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
            IdealThruster(time_constant=0.4,
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
    
    # Override Mixer
    # mixer = np.array([
    #     [0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0],
    #     [1.0, 1.0, 1.0, 1.0],
    #     [0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0],
    #     [3.0, -3.0, 3.0, -3.0],
    #     [0.0, 0.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 0.0],
    # ])
    
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

    # K = np.array([
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # x'
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # y'
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # z'
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # x''
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # y''
    #                     [0, 0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # z''
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # phi'
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # theta'
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # psi'
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # phi''
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # theta''
    #                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],          # psi''
    #                       dtype=float)
    
    K = mixer.transpose() @ -K

    # print(K)

    fc = FlightComputer(
        sample_time=0.001,
        sensors=[],
        K=BlizzardController(),
    )

    # fc=FlightComputer(
    #         sample_time=0.001,
    #         sensors=[],
    #         K=K,
    #         # K = np.zeros((12,4))
    #     )


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
        recording_sample_time=0.1,
        max_recorder_buffer_steps=1000,
        run_name="nn_train_att_1"
    )

    return blizzard

def blizzard_hover_scenario_setup(name=""): 
    
    environments = [
        ConstantField(),
        # PointMassGravity(mass=5.97219E24, position=np.array([0.0, 0.0, -6.378E6]))
    ]

    blizzard_instance = Object_Instance(
        "blizzard",
        np.zeros((12,1)),
        "ForwardEuler",
        8,
    )

    # object_list = [blizzard_instance for i in range(10)]
    # print(object_list)
    object_list = [blizzard_instance]

    return Scenario(scenario_name="blizzard_hover_test", end_time=15.0, min_dt=0.001, objects=object_list, environments=environments)

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

    # object_list = [blizzard_instance for i in range(10)]
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

def train_model(job, model, optimizer, loss_function, num_epochs=100, noise_scale=0.1, num_variations=5):
    input_dim = 12  # State vector size
    onnx_path = job.path + "/objects/blizzard/blizzard.onnx"

    best_loss = float("inf")
    best_model_params = None

    for epoch in range(num_epochs):
        

        # Try multiple weight variations
        for _ in range(num_variations):
            # Perturb weights slightly to explore new policies
            with torch.no_grad():
                for param in model.parameters():
                    param += noise_scale * torch.randn_like(param)
            job.export_job()
            os.system("./target/release/sixdof")
            import analyzedata
            sim_output = SimObjectOutput.read_from_csv("data/todo/default_name/output/blizzard_hover_test/object_0_blizzard_0.csv")
        

            # Load simulation results
            loss_value = compute_trajectory_loss(sim_output)

            # Keep track of the best policy
            if loss_value < best_loss:
                best_loss = loss_value
                best_model_params = {k: v.clone() for k, v in model.state_dict().items()}

        # Update model weights to the best-performing one
        if best_model_params:
            model.load_state_dict(best_model_params)

        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Best Loss: {best_loss:.4f}")
            # analyze_scenario("data/todo/default_name/output/blizzard_hover_test")

def train_evolution_strategy(job, model, num_generations=50, population_size=10, mutation_scale=0.1):
    input_dim = 12  # State vector size
    # onnx_path = job.path + "/objects/blizzard/blizzard.onnx"

    best_model_params = {k: v.clone() for k, v in model.state_dict().items()}
    best_loss = float("inf")

    for generation in range(num_generations):
        population = []

        # Generate population by mutating the best model
        for _ in range(population_size):
            new_model = BlizzardController()
            new_model.load_state_dict(best_model_params)

            with torch.no_grad():
                for param in new_model.parameters():
                    param += mutation_scale * torch.randn_like(param)

            population.append(new_model)

        # Evaluate population
        for candidate_model in population:
            job.objects[0].fc.K.network = candidate_model
            job.export_job()
            os.system("./target/release/sixdof")
            # import analyzedata
            sim_output = SimObjectOutput.read_from_csv("data/todo/default_name/output/blizzard_hover_test/object_0_blizzard_0.csv")
            loss_value = compute_trajectory_loss(sim_output)

            if loss_value < best_loss:
                best_loss = loss_value
                best_model_params = {k: v.clone() for k, v in candidate_model.state_dict().items()}

        print(f"Generation {generation}, Best Loss: {best_loss:.4f}")

    # Load the best model
    # model.load_state_dict(best_model_params)

def basic_job():
    job = Job()

    job.add_object(blizzard_test_object_setup())
    job.add_scenario(blizzard_hover_scenario_setup())
    [job.add_scenario(blizzard_attitude_stabilization_setup(f"attitude_test_{i}")) for i in range(5)]

    # job.add_scenario(test_scenario())
    # job.add_scenario(test_quadcopter())
    # job.add_scenario()
    # test_scenario()
    # test_quadcopter()
    # test_constellation_scenario()

    job.export_job()

    # dummy_input = torch.zeros(1,12)
    # controller = BlizzardController()
    # torch.onnx.export(
    #     controller,
    #     dummy_input,
    #     "data/todo/default_name/objects/blizzard/blizzard_controller.onnx",
    #     export_params=True,
    #     opset_version=11,
    #     do_constant_folding=True,
    #     input_names=['input'],
    #     output_names=['output']
    # )

if __name__ == "__main__":
    job = nn_train_att_1()


    model = job.objects[0].fc.K
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train_model(job, model, optimizer, compute_trajectory_loss, num_epochs=100, noise_scale=0.1, num_variations=5)
    train_evolution_strategy(job, model, num_generations=50, population_size=10, mutation_scale=5.0)
    # basic_job()

