pub type State = na::SMatrix<f64, 12, 1>;
pub type Inputs<const U: usize> = na::SMatrix<f64, U, 1>; 

pub const GRAPHICAL_SCALING_FACTOR: f64 = 1E0;
