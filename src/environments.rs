use crate::datatypes;

pub struct OrbitalBody {
    mass: f64,
    radius: f64,
    atmosphere: Atmosphere,
    soi_radius: f64,
    
}

pub struct Atmosphere {
    atmosphere: u64,
}