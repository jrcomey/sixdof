use crate::datatypes::*;
use nalgebra as na;
use na::{Dyn, SMatrix};
use serde_json::{json, Value};
use crate::sixdof::Integrator;
use crate::graphical;
use std::f64::consts::PI;


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
        self.force = self.set_force;
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
