use na::SMatrix;

/// Number of states for 6DOF.
pub type State = na::SMatrix<f64, 12, 1>;

/// Datatype for inputs, dependent on vehicle
pub type Inputs<const U: usize> = na::SMatrix<f64, U, 1>; 

/// Scaling factor for graphics
pub const GRAPHICAL_SCALING_FACTOR: f64 = 1E0;

/// Limit value to +/- max_magnitude range
pub fn limit_value_symmetric(value: f64, max_magnitude: f64) -> f64 {
    
    // Reduce check if in range
    if value < max_magnitude && value > -max_magnitude {
        value
    } else if value > max_magnitude {
        max_magnitude
    } else  {
        -max_magnitude
    }
}

pub fn limit_value(value: f64, upper_limit: f64, lower_limit:f64) -> f64 {
    if value > upper_limit {
        upper_limit
    } else if value < lower_limit {
        lower_limit
    } else {
        value
    }
}

pub fn rotation_frame(phi: &f64, theta: &f64, psi: &f64) -> SMatrix<f64, 12, 12> {

    let mut R_big = SMatrix::<f64, 12, 12>::zeros();

    let R = na::SMatrix::<f64, 3, 3>::from_row_slice(&[
        psi.cos()*theta.cos(), psi.cos()*theta.sin()*phi.sin()-psi.sin()*phi.cos(), psi.cos()*theta.sin()*phi.cos()+psi.sin()*phi.sin(),
        psi.sin()*theta.cos(), psi.sin()*theta.sin()*phi.sin()+psi.cos()*phi.cos(), psi.sin()*theta.sin()*phi.cos()-psi.cos()*phi.sin(),
        -1.0*theta.sin(), theta.cos()*phi.sin(), theta.cos()*phi.cos()
    ]);

    for i in 0..=3 {
        R_big.view_mut((i*3,i*3), (3,3)).copy_from(&R);
        
    }
    

    return R_big
}