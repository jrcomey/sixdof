/*  TODO:
    // Indicates completed
    - Flight Computer and electronics
        - Basic stabilizing flight computer
        - Betaflight duplicate
        - 
    - Scenario Loading
        - Load Environments (Empty Space, Gravity, Etc)
        - Load Objects
        - Load Terrain (If Applicable)
    - Simulation Running
        - Skip static objects
        - Midpoint time steps for 


*/


#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]
#![allow(redundant_semicolons)]
#![allow(unused_assignments)]
#![allow(unreachable_patterns)]
#![allow(unused_mut)]
#![allow(unused_parens)]
#![allow(unused_variables)]
#![allow(unreachable_code)]

extern crate nalgebra as na;
mod sixdof;
extern crate pretty_env_logger;                                             // Logger
#[macro_use] extern crate log;
use std::error::Error;

// Logging crate
use crate::sixdof::Simulatable;
extern crate serde_json as sj;
extern crate serde;
extern crate indicatif;
mod setup;
mod datatypes;
mod environments;
mod fc;
mod graphical;
mod components;
fn main() -> Result<(), std::io::Error> {
    let t_start = std::time::Instant::now();
    std::env::set_var("RUST_LOG", "trace");                                 // Initialize logger
    pretty_env_logger::init();
    info!("Program Start!");

    let mut sim = sixdof::Sim::new(1.0E-3);
    
    let mut drone = setup::test_falling_object();


    
    sim.add_object(drone);
    sim.add_object(setup::static_cube());

    sim.datacom_start("127.0.0.1:8080").expect("No connectioned established.");
    sim.run_until(1.0).expect("Sim failed!");
    // // println!("{}", sim.scene_initialization_to_datacom());
    info!("Program End!");
    info!("Final Position: {}", sim.get_object(0).get_position());
    // sim.record_run("data/runs");
    // info!("Run recorded. Saved to data/runs");
    // println!("Program finished in {:.2?} in {:} steps", t_start.elapsed(), sim.steps);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn RK4_gravity_test_big_dt() {
        let mut sim = sixdof::Sim::new(0.1);
        let mut drone = Box::new(sixdof::Vehicle::<4>::new());
        let A_new = na::SMatrix::<f64, 12, 12>::from_row_slice(&[
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let state_new = na::SMatrix::<f64, 12, 1>::from_row_slice(&[
            0.0,    // x
            0.0,    // y
            0.0,    // z
            0.0,    // xdot
            0.0,    // ydot
            0.0,    // zdot
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]);
        let C_new = na::SMatrix::<f64, 6, 12>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ]);
        drone.set_A(
            A_new
        );

        // let Rocket = Box::new(sixdof::Rocket::new());
        drone.set_state(state_new);
        sim.add_object(drone);
        let t = 10.0;
        sim.run_until(t);
        let final_position = sim.get_object(0).get_position()[2];
        let final_predicted = (1.0/2.0) * t*t * (-9.81);

        assert!(0.99*final_predicted > final_position);
        assert!(1.01*final_predicted < final_position);
    }

    #[test]
    fn RK4_gravity_test_small_dt() {
        let mut sim = sixdof::Sim::new(0.001);
        let mut drone = Box::new(sixdof::Vehicle::<4>::new());
        let A_new = na::SMatrix::<f64, 12, 12>::from_row_slice(&[
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let state_new = na::SMatrix::<f64, 12, 1>::from_row_slice(&[
            0.0,    // x
            0.0,    // y
            0.0,    // z
            0.0,    // xdot
            0.0,    // ydot
            0.0,    // zdot
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]);
        let C_new = na::SMatrix::<f64, 6, 12>::from_row_slice(&[
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ]);
        drone.set_A(
            A_new
        );

        // let Rocket = Box::new(sixdof::Rocket::new());
        drone.set_state(state_new);
        sim.add_object(drone);
        let t = 10.0;
        sim.run_until(t);
        let final_position = sim.get_object(0).get_position()[2];
        let final_predicted = (1.0/2.0) * t*t * (-9.81);

        assert!(0.9999*final_predicted > final_position);
        assert!(1.0001*final_predicted < final_position);
    }

    #[test]
    fn uncontrolled_position_input() {
        let mut sim = sixdof::Sim::new(1.0E-3);
        let drone = setup::non_falling_obect();
        sim.add_object(drone);
        let t = 10.0;
        sim.run_until(t);
        let final_position = sim.get_object(0).get_position()[2];
        let final_unassisted = (1.0/2.0) * t*t * (-9.81);
        assert!(final_position > 0.5*final_unassisted);
    }


}
