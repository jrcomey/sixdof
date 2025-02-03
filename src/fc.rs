use std::vec;

use sj::Value;
use tract_onnx::prelude::*;

use crate::datatypes::*;

pub struct FlightComputer<const U: usize> {
    sample_time: f64,
    t_last_updated: f64,
    sensors: Vec<Sensor>,
    cmd_inputs: Inputs<U>,
    K: na::SMatrix<f64, U, 12>,
}

pub trait FlightControl<const U: usize> {
    fn estimate_state(&mut self, actual_state: State) -> State;
    fn guidance(&mut self, t: &f64, estimated_state: State) -> State;
    fn calculate_u(&self, current_state: State) -> Inputs<U>;
}

impl<const U: usize> FlightComputer<U> {
    pub fn new(sample_time: f64, sensors: Vec<Sensor>, K: na::SMatrix<f64, U, 12>) -> FlightComputer<U> {
        FlightComputer {
            sample_time: sample_time,
            t_last_updated: 0.0,
            sensors: sensors,
            cmd_inputs: Inputs::zeros(),
            K: K
        }
    }

    pub fn new_from_json(json_parsed: &Value) -> Self {
        
        let sample_time = match &json_parsed["sample_time"] {
            Value::Number(t_s) => {t_s.as_f64().unwrap()},
            _ => {0.0}
        };

        // Unimplemented
        let sensors = vec![];

        let K_dat: Vec<f64> = json_parsed["K"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|x| x.as_f64().unwrap())
            .collect();

        let K: na::SMatrix<f64, U, 12> = na::SMatrix::from_row_slice(&K_dat[..]);

        FlightComputer{
            sample_time: sample_time,
            t_last_updated: 0.0,
            sensors: sensors,
            cmd_inputs: Inputs::zeros(),
            K: K
        }
        
    }
}

impl<const U: usize> FlightControl<U> for FlightComputer<U> {
    fn estimate_state(&mut self, actual_state: State) -> State {
        return actual_state;
    }

    fn calculate_u(&self, current_state: State) -> Inputs<U> {
        // debug!("FC OUTPUT: {}", self.K*current_state);
        let err: State = rotation_frame(&current_state[6], &current_state[7], &current_state[8]).transpose()
        * (State::zeros() - current_state);
        // debug!("K: {}", self.K);
        // debug!("FC OUTPUT: {}", -self.K*err);
        -self.K*err
    }

    fn guidance(&mut self, t: &f64, estimated_state: State) -> State {
        State::zeros()
    }
}

pub struct NullComputer<const U: usize> {
    inputs: Inputs<U>,
}

impl<const U: usize> NullComputer<U> {
    pub fn new() -> NullComputer<U> {
        NullComputer{
            inputs: Inputs::<U>::zeros()
        }
    }
}

impl<const U: usize> FlightControl<U> for NullComputer<U> {
    
    fn estimate_state(&mut self, actual_state: State) -> State {
        return State::zeros();
    }

    fn guidance(&mut self, t: &f64, estimated_state: State) -> State {
        State::zeros()
    }

    fn calculate_u(&self, current_state: State) -> Inputs<U> {
        return Inputs::<U>::zeros();
    }
}

pub struct NNComputer<const U: usize> {
    sample_time: f64,
    t_last_updated: f64,
    sensors: Vec<Sensor>,
    cmd_inputs: Inputs<U>,
    network: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl<const U: usize> NNComputer<U> {

    fn new_from_file(json_parsed: &Value) -> Result<Self, std::io::Error> {
        let model = tract_onnx::onnx()
        .model_for_path("data/todo/default_name/objects/blizzard/blizzard_controller.onnx").unwrap()
        .into_optimized().unwrap()
        .into_runnable().unwrap();

        let nn_computer = NNComputer {
            sample_time: 0.0,
            t_last_updated: 0.0,
            sensors: vec![],
            cmd_inputs: Inputs::zeros(),
            network: model,
        };
        Ok(nn_computer)
    }


}

impl<const U: usize> FlightControl<U> for NNComputer<U>{
    
    fn estimate_state(&mut self, actual_state: State) -> State {
        actual_state   
    }

    fn guidance(&mut self, t: &f64, estimated_state: State) -> State {
        State::zeros()
    }

    fn calculate_u(&self, current_state: State) -> Inputs<U> {
        let err: State = rotation_frame(&current_state[6], &current_state[7], &current_state[8]).transpose()
        * (State::zeros() - current_state);

        let input = tract_ndarray::Array2::from_shape_fn(
            (1,12),
            |(i,j)| err[(i, j)]
        ).into_tensor();

        let result = self.network.run(tvec!(input.into())).unwrap();

        let output_view = result[0].to_array_view::<f64>().unwrap();
        let output = Inputs::from_row_slice(
            output_view.as_slice().unwrap()
        );

        return output
    }
    
}

// #####################


pub struct Sensor {
    null: u64,
}

impl Sensor {
    pub fn sense(&self, vehicle_state: State) -> State {
        return vehicle_state;
    }
}

pub trait NavComputer {
    fn estimate_state(&mut self, true_state: &State, sensors: Vec<Sensor>) -> State;
    // fn 
}