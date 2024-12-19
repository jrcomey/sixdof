use crate::{datatypes::*, sixdof};
use nalgebra as na;
use na::{Dyn, SMatrix};
use serde_json::{json, Value};
use crate::sixdof::Integrator;
use crate::graphical;
use std::f64::consts::PI;
// use serde_json::{json, Value};


pub trait ComponentPart {
    fn update_to_time(&mut self, y: Vec<f64>, t: f64);
    fn get_force_on_parent(&self) -> SMatrix<f64, 12, 1>;
    fn set_force(&mut self, new_force:f64);
    fn datacom_get_model_json(&self) -> Value;
}

pub struct ElectricalMotor {
    J: f64,                                         // Moment of Inertia [kg*m^2]
    Kv: f64,                                        // Back EMF constant
    Kt: f64,                                        // Motor Torque Constant
    R: f64,                                         // Motor Phase Resistance
    L: f64,                                         // Motor Phase Inductance
    N: f64,                                         // Number of pole pairs
    theta: f64,                                     // Motor Rotational Position [rad]
    omega: f64,                                     // Motor rotational speed
    relative_position: na::Vector3<f64>,            // Motor position relative to parent
    relative_rotation: na::Vector3<f64>,            // Motor rotation relative to parent
    i: f64,                                          // Current in the windings
    last_time: f64                                  // Last time
    // integrator: sixdof::Integrator
}

impl ElectricalMotor {
    pub fn new(J: f64, Kv: f64, Kt: f64, R: f64, L: f64, N: f64, relative_position: na::Vector3<f64>, relative_rotation: na::Vector3<f64>) -> ElectricalMotor {
        
        ElectricalMotor { 
            J: J,
            Kv: Kv,
            Kt: Kt,
            R: R,
            L: L,
            N: N,
            theta: 0.0,
            omega: 0.0,
            relative_position: relative_position,
            relative_rotation: relative_rotation,
            i: 0.0,
            last_time: 0.0,
            // integrator: Integrator::ForwardEuler,
        }
    }

    pub fn new_from_json() -> ElectricalMotor {
        todo!();
    }

    /// Limits rotation to bounds of [-PI, PI]
    /// 
    /// Floating point precision is higher closer to 0.0
    pub fn bounds_check(&mut self) {
        if self.theta < -PI {
            while self.theta < -PI {
                self.theta = self.theta + 2.0*PI;
            }
        }
        else if self.theta > PI {
            while self.theta > PI {
                self.theta = self.theta - 2.0*PI;
            }
        }
    }

    fn calculate_back_emf(&self) -> f64 {
        self.Kt*self.omega
    }

    fn calculate_torque(&mut self, target_speed: f64, voltage: f64, t: f64) -> na::Vector3<f64> {
        // Calculate dt
        let dt = t - self.last_time;
        self.last_time = t;
        
        // Electrical dynamics
        // di/dt = (V - Ke*omega - R*i)/L
        let back_emf = self.Kv * self.omega;
        let di_dt = (voltage - back_emf - self.R * self.i) / self.L;
        self.i += di_dt * dt;
        
        // Mechanical dynamics
        // dω/dt = (Kt*i - B*ω)/J
        // Using a simple viscous damping model where B is proportional to speed
        let damping = 0.1 * self.omega;  // Simplified damping coefficient
        let torque = self.Kt * self.i;
        let domega_dt = (torque - damping) / self.J;
        
        let rotation_matrix = na::Rotation3::from_euler_angles(self.relative_rotation[0], self.relative_rotation[1], self.relative_rotation[2]);
        let torque_vec = rotation_matrix * na::Vector3::new(torque, 0.0, 0.0);

        // Update motor speed and position
        self.omega += domega_dt * dt;
        self.theta += self.omega * dt;
        
        return torque_vec
    }
}

impl ComponentPart for ElectricalMotor {
    fn datacom_get_model_json(&self) -> Value {
        todo!()
    }

    fn get_force_on_parent(&self) -> SMatrix<f64, 12, 1> {
        // Calculate thrust force (simplified model assuming propeller)
        // F = Kt * i * omega for crude approximation
        let thrust_magnitude = self.Kt * self.i * self.omega.abs();
        
        // Convert thrust to body frame using relative rotation
        // Assuming relative_rotation contains roll, pitch, yaw angles
        let rotation_matrix = na::Rotation3::from_euler_angles(self.relative_rotation[0], self.relative_rotation[1], self.relative_rotation[2]);
        let thrust_vector = rotation_matrix * na::Vector3::new(thrust_magnitude, 0.0, 0.0);
        
        let parent_force = na::SMatrix::<f64, 12, 1>::from_row_slice(&[
            0.0,
            0.0,
            0.0,
            thrust_vector[0],
            thrust_vector[1],
            thrust_vector[2],
            0.0,
            0.0,
            0.0,
            thrust_vector[0]*self.relative_position[0],
            thrust_vector[1]*self.relative_position[1],
            thrust_vector[2]*self.relative_position[2],
        ]);
        return parent_force
    }

    fn set_force(&mut self, new_force:f64) {
        todo!();
    }

    fn update_to_time(&mut self, y: Vec<f64>, t: f64) {
        todo!();
    }
}

pub struct RocketMotor {
    ignited: IgnitionStatus,
    time_since_ignition: f64,
    burn_time_vec: Vec<f64>,
    thrust_curve: Vec<f64>
}

impl RocketMotor {

    pub fn new(burn_time_vec: Vec<f64>, thrust_curve: Vec<f64>) -> RocketMotor {

        RocketMotor {
            ignited: IgnitionStatus::Ready,
            time_since_ignition: 0.0,
            burn_time_vec: vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            thrust_curve: vec![1.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 2.5, 0.0]
        }
    }
}

enum IgnitionStatus {
    Ready,
    Active,
    Spent
}

// ###################

pub struct IdealThruster {
    time_const: f64,
    force: f64,
    set_force: f64,
    last_time: f64,
    position: na::SMatrix<f64, 3, 1>,
    orientation: na::UnitVector3<f64>,
    integrator: Integrator,
    model: graphical::GraphicalData,
}

impl IdealThruster {

    pub fn new() -> Self {
        IdealThruster {
            time_const: 0.0001,
            force: 0.0,
            set_force: 0.0,
            last_time: 0.0,
            position: na::SMatrix::zeros(),
            orientation: na::UnitVector3::new_unchecked(na::Vector3::new(0.0, 0.0, 1.0)),
            integrator: Integrator::RK4,
            ..Default::default()
        }
    }

    pub fn set_force(&mut self, new_force: f64) {
        self.set_force = new_force;
    }

    pub fn get_xdot(&self, y: &na::SMatrix<f64, 1, 1>, t: &f64) -> na::SMatrix<f64, 1, 1>{
        return (na::SMatrix::<f64, 1, 1>::new(self.force) - y)/self.time_const;
    }

    pub fn set_time_const(&mut self, new_time_const: f64) {
        self.time_const = new_time_const;
    }

    pub fn get_force(&self) -> f64 {
        self.force
    }

    pub fn load_from_json_parsed(json_parsed: Value) -> Self {

        let time_const = json_parsed["time_constant"].as_f64().unwrap();
        let position_vec: Vec<f64> = json_parsed["position"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let position = na::Vector3::new(position_vec[0], position_vec[1], position_vec[2]);

        let orientation_vec: Vec<f64> = json_parsed["orientation"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let orientation = na::Vector3::new(orientation_vec[0], orientation_vec[1], orientation_vec[2]);
        let orientation = na::UnitVector3::new_normalize(orientation);

        IdealThruster {
            time_const: time_const,
            force: 0.0,
            set_force: 0.0,
            last_time: 0.0,
            position: position,
            orientation: orientation,
            integrator: Integrator::ForwardEuler,
            ..Default::default()
        }
    }
}

impl ComponentPart for IdealThruster {
    fn get_force_on_parent(&self) -> State {
        let force_vec = self.force * *self.orientation;
        let torque_vec = force_vec.cross(&self.position);
        let force_and_torque = State::from_row_slice(&[
            0.0, 
            0.0,
            0.0,
            force_vec[0],
            force_vec[1],
            force_vec[2],
            0.0,
            0.0,
            0.0,
            torque_vec[0],
            torque_vec[1],
            torque_vec[2]
        ]);
        return force_and_torque;
    }

    fn update_to_time(&mut self, y: Vec<f64>, t: f64) {
        let dt = t-self.last_time;
        // let force_set = y[0];
        let force_at_time_step = (self.set_force + (self.force- self.set_force) * (-1.0*dt/self.time_const).exp());

        // self.force = self.set_force;

        self.force = force_at_time_step;
    }

    fn set_force(&mut self, new_force: f64) {
        self.set_force = new_force;
    }

    fn datacom_get_model_json(&self) -> Value {
        self.model.get_model_value()
    }
}

impl Default for IdealThruster {
    fn default() -> Self {
        IdealThruster {
            time_const: 0.0001,
            force: 0.0,
            set_force: 0.0,
            last_time: 0.0,
            position: na::SMatrix::zeros(),
            orientation: na::UnitVector3::new_unchecked(na::Vector3::new(0.0, 0.0, 1.0)),
            integrator: Integrator::RK4,
            model: graphical::GraphicalData{..Default::default()}
        }
    }
}
