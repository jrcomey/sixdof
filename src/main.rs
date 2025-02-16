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
        // - Skip static objects
        - Midpoint time steps for digital applications

    - Simulation capabilities
        - create functions to change rotation frames
        - Datalogger should check if output folder is empty at sim start and clear it if not

*/

#![allow(warnings)]
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
use crate::components::ComponentPart;
extern crate tract_onnx;
// use tract_onnx::prelude::*;
// use tract_ndarray::Array;
fn main() -> Result<(), std::io::Error> {
    let t_start = std::time::Instant::now();
    std::env::set_var("RUST_LOG", "sixdof=trace,warn cargo run");                                 // Initialize logger
    pretty_env_logger::init();
    info!("Program Start!");

    sixdof::Sim::load_and_run_job("data/todo").unwrap();

    // let mut sim = sixdof::Sim::load_scenario("data/todo/default_name/scenarios/default_scenario_name.json");
    // sim.datacom_start("10.0.0.107:8080").expect("No connectioned established.");
    // // sim.datacom_start("127.0.0.1:8080").expect("No connection established.");       // Localhost
    // // sim.run_until(30.0*5400.0).expect("Sim failed!");
    // sim.run();
    // // // println!("{}", sim.scene_initialization_to_datacom());
    // info!("Program End!");
    // // let name = sim.get_object(0).get_name();
    // info!("Final Position: {}", sim.get_object(0).get_position());
    // println!("Program finished in {:.2?} in {:} steps", t_start.elapsed(), sim.steps);
    // // let filepath = format!("/data/runs/{}", sim.)
    // sim.record_run("data/runs/test_drone");
    // info!("Run recorded. Saved to data/runs");

    Ok(())
}

#[cfg(test)]
mod tests {
    use core::time;

    use components::{ComponentPart, IdealThruster};
    use datatypes::*;

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

        println!("Final Position: {}", final_position);
        println!("Predicted Position: {}", final_predicted);

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

        println!("Final Position: {}", final_position);
        println!("Predicted Position: {}", final_predicted);

        assert!(0.99*final_predicted > final_position);
        assert!(1.01*final_predicted < final_position);
    }

    #[test]
    fn uncontrolled_position_input() {
        std::env::set_var("RUST_LOG", "trace");                                 // Initialize logger
        pretty_env_logger::init();
        let mut sim = sixdof::Sim::new(1.0E-3);
        let drone = setup::non_falling_obect();
        sim.add_object(drone);
        let t = 10.0;
        sim.run_until(t);
        let final_position = sim.get_object(0).get_position()[2];
        let final_unassisted = (1.0/2.0) * t*t * (-9.81);
        trace!("Final Position: {}", final_position);
        trace!("Position with no input: {}", final_unassisted);
        assert!(final_position > 0.5*final_unassisted);
    }

    #[test]
    fn simple_thruster_step_response() {
        let mut thruster = IdealThruster::new();
        let time_vec = linspace(0.0, 10.0, 100);
        let result_vec: Vec<f64> = vec![];
        thruster.set_time_const(0.01);

        thruster.set_force(1.0);
        for t in time_vec {
            thruster.update_to_time(vec![0.0], t);
            debug!("Time: {}", t);
            debug!("Force: {}", thruster.get_force_on_parent()[3]);
        }
    }

    #[test]
    fn rotation_null() {
        let R = rotation_frame(&0.0, &0.0, &0.0);
        // println!("R: {}", R);
        assert_eq!(R, na::SMatrix::<f64, 12, 12>::identity());
    }

    #[test]
    fn rotate_psi() {
        let R = rotation_frame(&0.0, &0.0, &(std::f64::consts::PI*0.5));
        let mut init_state = State::zeros();
        init_state[0] = 1.0;
        let new_state = R * init_state;
        let mut comparison_state = State::zeros();
        comparison_state[1] = 1.0;
        println!("New State: {}", new_state);
        // assert!();
    }

    #[test]
    fn rotate_transpose() {
        let R = rotation_frame(&0.0, &0.0, &(std::f64::consts::PI*0.5));
        let mut init_state = State::zeros();
        init_state[0] = 1.0;


        assert_eq!(init_state, R * R.transpose() * init_state);
    }

}

/// Linspace function
fn linspace(start: f64, end: f64, num_points: usize) -> Vec<f64> {
    let step = (end - start) / (num_points as f64 - 1.0);
    (0..num_points)
        .map(|i| start + (i as f64) * step)
        .collect()
}