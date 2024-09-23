use std::default;

use crate::datatypes;
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
            grav_field: GravitationalField{..Default::default()},
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

pub struct GravitationalField {
    mass: f64,
    soi_radius: f64,
}

impl Default for GravitationalField {
    fn default() -> Self {
        GravitationalField {
            mass: 0.0, 
            soi_radius: 1.0E3
        }
    }
}