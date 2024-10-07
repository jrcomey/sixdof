import json
import numpy as np

class Scenario:

    def __init__(self, environments = []):
        pass

    def add_object(self, object):
        pass

    def add_environment(self, environment):
        pass

    def write_run_to_file(self):
        pass

class Object:

    def __init__():
        pass

    def create_json_dict():
        data = {
            "default": 0,
        }

        return data

    def to_json_str(filename):
        return json.dumps(self.create_json_dict)
    



class Component:

    def __init__():
        pass

    def create_json_dict():
        data = {
            "default": 0,
        }

        return data

    def to_json_str(filename):
        return json.dumps(self.create_json_dict)

class Environment:

    def __init__(self):
        pass

    def create_json_dict(self):
        data = {
            "default": 0,
        }

        return data

    def to_json_str(filename):
        return json.dumps(self.create_json_dict)
    
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
            "position": self.position.to_list()
        }


# BASIC SCENARIO
    
def test_scenario():
    # Earth Environment
    environments = [
        PointMassGravity(mass=5.97219E24)
    ]