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
}

impl<const U: usize> FlightControl<U> for FlightComputer<U> {
    fn estimate_state(&mut self, actual_state: State) -> State {
        return actual_state;
    }

    fn calculate_u(&self, current_state: State) -> Inputs<U> {
        // debug!("{}", self.K*current_state);
        self.K*current_state
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
