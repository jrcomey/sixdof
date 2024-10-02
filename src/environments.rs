use std::default;

use crate::datatypes::*;
use na::Point;
use nalgebra as na;
use std::option::Option;


/// Universal gravitational constant. 
const GRAV_CONST_UNIV: f64 = 6.67430E-11; // N*m^2/ kg^2

pub struct OrbitalBody {
    grav_field: GravitationalField,
    atmosphere: Option<Atmosphere>,
    position: na::Point3<f64>
}

impl Default for OrbitalBody {
    fn default() -> Self {
        OrbitalBody{
            grav_field: GravitationalField::default(),
            atmosphere: Option::Some(Atmosphere {..Default::default()}),
            position: na::Point3::origin(),
        }
    }
}

pub struct Atmosphere {
    atmosphere: u64,
}   

impl Default for Atmosphere {
    fn default() -> Self {
        Atmosphere { atmosphere: 0 }
    }
}

/// Enum for gravitational field. Can either be a point mass, or a constant field.
#[derive(Clone, Copy)]
pub enum GravitationalField {
    PointMass {
        mass: f64,
        soi_radius: f64,
        position: na::Point3<f64>
    },
    Constant {
        acceleration: f64,
        direction: na::Vector3<f64>
    }
}

// pub struct PointGravitationalField {
//     mass: f64,
//         soi_radius: f64,
//         position: na::Point3<f64>
// }


// impl EnviromentalEffect for PointGravitationalField {

// }


// pub struct StaticGravitationalField {
//     acceleration: f64,
//     direction: na::Vector3<f64>
// }

// impl EnviromentalEffect for StaticGravitationalField {

// }

// impl Default for StaticGravitationalField {
//     fn default() -> Self {
//         StaticGravitationalField {
//             acceleration: 9.80665,
//             direction: -na::Vector3::z()
//         }
//     }
// }



impl GravitationalField {
    fn new_point_field(mass: f64, soi_radius: f64, position: na::Point3<f64>) -> Self {
        GravitationalField::PointMass { mass: mass, soi_radius: soi_radius, position: position }
    }

    fn new_static_field(acceleration: f64, direction: na::Vector3<f64>) -> Self {
        GravitationalField::Constant { acceleration: acceleration, direction: direction }
    }
}

impl Default for GravitationalField {
    fn default() -> Self {
        GravitationalField::Constant { acceleration: 9.80665, direction: -na::Vector3::z() }
    }
}





pub trait EnviromentalEffect {
    fn calculate_acceleration_on_object(&self, vehicle_mass: &f64, vehicle_position: &na::Point3<f64>) -> State;
}

impl EnviromentalEffect for GravitationalField {
    fn calculate_acceleration_on_object(&self, vehicle_mass: &f64, vehicle_position: &na::Point3<f64>) -> State {
        let x_dot_grav: State = match self {
            GravitationalField::PointMass { mass, soi_radius, position } => {
                let r: na::Vector3<f64> = vehicle_position - position;
                // debug!("Vehicle Position: {}", vehicle_position);
                // debug!("Point Mass Position: {}", position);
                let acc_vector = -GRAV_CONST_UNIV * mass  / (r.magnitude_squared()) * r.normalize();
                // let acc_vector = r/r.magnitude();
                // debug!("Magnitude of radius: {}", r.magnitude());
                // debug!("Point Mass Gravitational Acceleration: {}", acc_vector);
                State::from_row_slice(&[
                    0.0,                    // x_dot
                    0.0,                    // y_dot
                    0.0,                    // z_dot
                    acc_vector[0],          // x_ddot
                    acc_vector[1],          // y_ddot
                    acc_vector[2],          // z_ddot
                    0.0,                    // theta_dot
                    0.0,                    // phi_dot
                    0.0,                    // psi_dot
                    0.0,                    // theta_ddot
                    0.0,                    // phi_ddot
                    0.0,                    // psi_ddot
                ])
            },
            GravitationalField::Constant { acceleration, direction } => {
                debug!("CONSTANT FIELD USED");
                let acc_vector = (*acceleration)*direction;
                State::from_row_slice(&[
                    0.0,                    // x_dot
                    0.0,                    // y_dot
                    0.0,                    // z_dot
                    acc_vector[0],          // x_ddot
                    acc_vector[1],          // y_ddot
                    acc_vector[2],          // z_ddot
                    0.0,                    // theta_dot
                    0.0,                    // phi_dot
                    0.0,                    // psi_dot
                    0.0,                    // theta_ddot
                    0.0,                    // phi_ddot
                    0.0,                    // psi_ddot
                ])
            }
        };
        return x_dot_grav;
    }
}