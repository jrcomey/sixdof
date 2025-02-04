use std::vec;

use sj::Value;
use tract_onnx::{prelude::*};

use crate::datatypes::*;

pub struct FlightComputer<const U: usize> {
    sample_time: f64,
    t_last_updated: f64,
    sensors: Vec<Sensor>,
    cmd_inputs: Inputs<U>,
    K: na::SMatrix<f64, U, 12>,
    guidance_state: State,
}

pub trait FlightControl<const U: usize> {
    fn estimate_state(&mut self, actual_state: State) -> State;
    fn calculate_guidance(&mut self, t: &f64, estimated_state: State) -> State;
    fn calculate_u(&self, current_state: State) -> Inputs<U>;
    fn get_cmd_position(&self) -> State;
}

impl<const U: usize> FlightComputer<U> {
    pub fn new(sample_time: f64, sensors: Vec<Sensor>, K: na::SMatrix<f64, U, 12>) -> FlightComputer<U> {
        FlightComputer {
            sample_time: sample_time,
            t_last_updated: 0.0,
            sensors: sensors,
            cmd_inputs: Inputs::zeros(),
            K: K,
            guidance_state: State::zeros(),
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
            K: K,
            guidance_state: State::zeros()
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

    fn calculate_guidance(&mut self, t: &f64, estimated_state: State) -> State {
        State::zeros()
    }

    fn get_cmd_position(&self) -> State {
        self.guidance_state
    }
}

pub struct NullComputer<const U: usize> {
    inputs: Inputs<U>,
    guidance_state: State
}

impl<const U: usize> NullComputer<U> {
    pub fn new() -> NullComputer<U> {
        NullComputer{
            inputs: Inputs::<U>::zeros(),
            guidance_state: State::zeros()
        }
    }
}

impl<const U: usize> FlightControl<U> for NullComputer<U> {
    
    fn estimate_state(&mut self, actual_state: State) -> State {
        return State::zeros();
    }

    fn calculate_guidance(&mut self, t: &f64, estimated_state: State) -> State {
        State::zeros()
    }

    fn calculate_u(&self, current_state: State) -> Inputs<U> {
        return Inputs::<U>::zeros();
    }

    fn get_cmd_position(&self) -> State {
        self.guidance_state
    }
}

pub struct NNComputer<const U: usize> {
    sample_time: f64,
    t_last_updated: f64,
    sensors: Vec<Sensor>,
    cmd_inputs: Inputs<U>,
    network: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    guidance_state: State,
}

impl<const U: usize> NNComputer<U> {

    pub fn new_from_json(json_parsed: &Value) -> Result<Self, std::io::Error> {
        let model = tract_onnx::onnx()
        .model_for_path("data/todo/default_name/objects/blizzard/blizzard.onnx").unwrap()
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 12))).unwrap()
        .into_optimized().unwrap()
        .into_runnable().unwrap();

        let nn_computer = NNComputer {
            sample_time: 0.0,
            t_last_updated: 0.0,
            sensors: vec![],
            cmd_inputs: Inputs::zeros(),
            network: model,
            guidance_state: State::zeros(),
        };
        Ok(nn_computer)
    }


}

impl<const U: usize> FlightControl<U> for NNComputer<U>{
    
    fn estimate_state(&mut self, actual_state: State) -> State {
        actual_state   
    }

    fn calculate_guidance(&mut self, t: &f64, estimated_state: State) -> State {
        State::zeros()
    }

    fn calculate_u(&self, current_state: State) -> Inputs<U> {
        let err: State = rotation_frame(&current_state[6], &current_state[7], &current_state[8]).transpose()
        * (State::zeros() - current_state);

        let input: Tensor = tract_ndarray::Array2::<f32>::from_shape_fn(
            (1,12),
            |(i,j)| err[(j,i)] as f32
        ).into_tensor();

        // debug!("Input: {:?}", input);

        let result = self.network.run(tvec!(input.into())).unwrap();

        let output_view = result[0].to_array_view::<f32>().unwrap();

        // debug!("Output: {:?}", output_view);
        let output_f64: Vec<f64> = output_view.iter().map(|&x| x as f64).collect();
        let output = Inputs::from_row_slice(&output_f64) * 3.8E3;
        return output
    }

    fn get_cmd_position(&self) -> State {
        self.guidance_state
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