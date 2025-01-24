import json
import numpy as np
import pretty_errors
import os



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

    def create_run_json(self):
        scenario_json = {
            "scenarioName": self.scenario_name,
            "endTime": self.end_time,
            "dtMin": self.min_dt,
            "vehicles": [obj.create_json_dict() for obj in self.objects],
            "environments": [env.create_json_dict() for env in self.environments],
            "datacomPort": self.datacom_port,
        }
        # print(scenario_json)

        # Make directories for sims if they don't already exist
        if not os.path.isdir(input:=f"data/todo"):
            os.mkdir(input)
        if not os.path.isdir(output:=f"data/runs/{self.scenario_name}"):
            os.mkdir(output)

        with open(input+f"/scenario_setup.json", "w") as file:
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

    def create_json_dict(self):
        data = {
            "sample_time": self.sample_time,
            "K": sum(self.K.tolist(), [])
        }
        return data
    
    def to_json_str(self):
        return json.dumps(self.create_json_dict())
        



class Vehicle:

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
            "name": self.name,
            "mass": self.mass,
            "rot_inertia": sum(self.I.tolist(), []),
            "state": sum(self.state.tolist(),[]),
            "A": sum(self.A.tolist(), []),
            "B": sum(self.B.tolist(), []),
            "physicsType": self.physics_type,
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
    print(r_iss)
    print(np.sqrt(r_iss[0]**2+r_iss[1]**2+r_iss[2]**2))
    print(400.0E3+6.378E6)
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
    mass_sim_ISS = Vehicle(
        name="ISS", 
        mass=1.0, 
        # graphical_path="data/test_object/default_cube.obj", 
        physics_type="RK4",
        init_state= init_state,
        graphical_elements=[GraphicalElement("CubeModel", "data/test_object/default_cube.obj", [0,0,0], [0,0,0], [0,0,0], [1,0,0,1], [100E3*1E-6,100E3*1E-6,100E3*1E-6])],
        recording_sample_time=0.0,
        max_recorder_buffer_steps=int(99E10)
    )

    earth_graphical = Vehicle(
        name="earth",
        mass=0,
        # graphical_path = "data/test_object/default_sphere.obj",
        physics_type="Static",
        graphical_elements=[GraphicalElement("EarthModel", "data/test_object/default_sphere.obj", [0,0,0], [0,0,0], [0,0,0], [0,0,1,1], [6.378E6*1E-6,6.378E6*1E-6,6.378E6*1E-6])]
    )


    scene = Scenario(objects=[earth_graphical, mass_sim_ISS], environments=environments, min_dt=1E-3, end_time=2*60*90)
    scene.create_run_json()

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

    earth_graphical = Vehicle(
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
        print(r_iss)
        print(np.sqrt(r_iss[0]**2+r_iss[1]**2+r_iss[2]**2))
        print(400.0E3+6.378E6)
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
        mass_sim_ISS = Vehicle(
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

    drone = Vehicle(
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

    static_cube = Vehicle(
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

    terrain = Vehicle(
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
    scene.create_run_json()

if __name__ == "__main__":
    # test_scenario()
    # test_quadcopter()
    test_constellation_scenario()