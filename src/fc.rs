use sj::Value;

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
        let err: State = State::zeros() - current_state;
        self.K*err
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

    fn calculate_u(&self, current_state: State) -> Inputs<U> {
        return Inputs::<U>::zeros();
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
