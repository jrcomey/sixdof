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
    guidance_computer: Box<dyn GuidanceComputer>,
}

pub trait FlightControl<const U: usize> {
    fn estimate_state(&mut self, actual_state: State) -> State;
    fn calculate_guidance(&mut self, t: &f64, estimated_state: State) -> State;
    fn calculate_u(&mut self, t: f64, current_state: State) -> Inputs<U>;
    fn get_cmd_position(&self) -> State;
}

impl<const U: usize> FlightComputer<U> {
    pub fn new(sample_time: f64, sensors: Vec<Sensor>, K: na::SMatrix<f64, U, 12>, guidance_computer: Box<dyn GuidanceComputer>) -> FlightComputer<U> {
        FlightComputer {
            sample_time: sample_time,
            t_last_updated: 0.0,
            sensors: sensors,
            cmd_inputs: Inputs::zeros(),
            K: K,
            guidance_state: State::zeros(),
            guidance_computer: guidance_computer,
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

        let guidance_computer: Box<dyn GuidanceComputer> = match json_parsed["guidanceType"].as_str().unwrap() {
            "TimingGuidance" => {
                Box::new(TimingGuidanceComputer::load_from_json(json_parsed["guidanceComputer"].clone()))
            },
            _ => {
                Box::new(ZeroGuidance::new())
            }
        };

        FlightComputer{
            sample_time: sample_time,
            t_last_updated: 0.0,
            sensors: sensors,
            cmd_inputs: Inputs::zeros(),
            K: K,
            guidance_state: State::zeros(),
            guidance_computer: guidance_computer,
        }
        
    }
}

impl<const U: usize> FlightControl<U> for FlightComputer<U> {
    fn estimate_state(&mut self, actual_state: State) -> State {
        return actual_state;    
    }

    fn calculate_u(&mut self, t: f64, current_state: State) -> Inputs<U> {
        // debug!("FC OUTPUT: {}", self.K*current_state);
        let err: State = rotation_frame(&current_state[6], &current_state[7], &current_state[8]).transpose()
        * (self.calculate_guidance(&t, current_state) - current_state);
        // debug!("K: {}", self.K);
        // debug!("FC OUTPUT: {}", -self.K*err);
        -self.K*err
    }

    fn calculate_guidance(&mut self, t: &f64, estimated_state: State) -> State {
        // State::zeros()
        self.guidance_computer.calculate_guidance(t, &estimated_state)
    }

    fn get_cmd_position(&self) -> State {
        self.guidance_computer.get_cmd_position()
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

    fn calculate_u(&mut self, t: f64, current_state: State) -> Inputs<U> {
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

    fn calculate_u(&mut self, t: f64, current_state: State) -> Inputs<U> {
        let err: State = (State::zeros() - current_state);

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

pub trait GuidanceComputer {
    fn calculate_guidance(&mut self, t: &f64, estimated_state: &State) -> State;
    fn get_cmd_position(&self) -> State;
}

pub struct TimingGuidanceComputer {
    t: f64,
    i: usize,
    next_transition_time: f64,
    current_target_state: State,
    transition_times: Vec<f64>,
    target_states: Vec<State>
}

impl TimingGuidanceComputer {
    pub fn new(transition_times: Vec<f64>, target_states: Vec<State>) -> Self {
        assert_eq!(
            transition_times.len(),
            target_states.len()
        );

        let next_time = if transition_times.len() > 1 {
            transition_times[1].clone()
        } else {
            transition_times[0].clone()
        };

        debug!("Next transition time: t={}", next_time);

        TimingGuidanceComputer {
            t: 0.0,
            i: 0,
            next_transition_time: next_time,
            current_target_state: target_states[0].clone(),
            transition_times: transition_times,
            target_states: target_states,
        }
    }

    pub fn load_from_json(json_parsed: Value) -> Self {

        let mut target_states: Vec<State> = match &json_parsed["targetStates"] {
            Value::Array(data_vector) => {
                let mut states = vec![];
                for data in data_vector{
                    let x_dat: Vec<f64> = data
                        .as_array()
                        .unwrap()
                        .into_iter()
                        .map(|x| x.as_f64().unwrap())
                        .collect();
                    let x: na::SMatrix<f64, 12, 1> = na::SMatrix::from_row_slice(&x_dat[..]);
                    states.push(x);
                }
                states
            },
            _ => {
                warn!("Tried to load target states but failed");
                vec![State::zeros()]
            }
        };

        let transition_times: Vec<f64> = json_parsed["transitionTimes"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
 
        Self::new(transition_times, target_states)
    }
}

impl GuidanceComputer for TimingGuidanceComputer {
    fn calculate_guidance(&mut self, t: &f64, estimated_state: &State) -> State {
        // debug!("Number of guidance points: {}", self.transition_times.len());
        self.t = t.clone();
        if self.i != self.target_states.len()-1 && self.t > self.next_transition_time {
            debug!("Advanced guidance state @ t={}", t);
            self.i += 1;
            if self.i != self.target_states.len()-1 {
                self.next_transition_time = self.transition_times[self.i+1];  // Come back to this one $FIXME    
            };
            self.current_target_state = self.target_states[self.i];
            debug!("Next time: {}", self.next_transition_time);
            return self.current_target_state
        } else {
            return self.current_target_state
        }
    }

    fn get_cmd_position(&self) -> State {
        // debug!("Current Target State: {}", self.current_target_state);
        return self.current_target_state
    }
}

/// Guidance Computer that always returns zero
pub struct ZeroGuidance {
    t: f64
}

impl ZeroGuidance {
    pub fn new() -> Self {
        ZeroGuidance { t: 0.0 }
    }
}

impl GuidanceComputer for ZeroGuidance {
    fn calculate_guidance(&mut self, t: &f64, estimated_state: &State) -> State {
        State::zeros()
    }

    fn get_cmd_position(&self) -> State {
        State::zeros()
    }
}

pub struct WaypointGuidanceComputer {

}