use core::{num, time};
use std::cell::RefCell;
use std::fmt::Error;
use std::sync::RwLock;
use std::time::{Duration, Instant};
use std::{default, u64};
use std::{sync::atomic::{AtomicU64, Ordering}};
use std::{fmt::format, mem::zeroed, vec};
use std::fs;
use std::path::Path;
// use std::io::Result;
use indicatif::ProgressBar;
use na::{Dyn, SMatrix};
use nalgebra as na;
use serde::de::value;
use serde::{Serialize, Deserialize};
use serde_json::{json, Value};
use tract_onnx::prelude::tract_itertools::enumerate;
use crate::datatypes::{rotation_frame, Inputs, State, GRAPHICAL_SCALING_FACTOR};
use crate::{environments, fc::*};
use std::net::TcpStream;
use std::io::{Read, Write};
use crate::graphical::{self, GraphicalData};
use std::fs::File;
use crate::components::*;
use crate::fc;
use rayon::prelude::*;
use std::sync::Arc;
use std::io;
// use std::path::Path;

/// Data transmission constant for chunk size, set at 4kB
const CHUNK_SIZE: usize = 4096;

#[derive(Clone)]
struct ScenarioInfo {
    job_name: String,
    scenario_name: String,
}

impl ScenarioInfo {
    fn new(job_name: String, scenario_name: String) -> Self {
        ScenarioInfo {
            job_name: job_name,
            scenario_name: scenario_name
        }
    }

    fn get_job_name(&mut self) -> &str {
        self.job_name.as_str()
    }
}

/// Top level sim struct, which contains objects to be simulated and their environments.
/// 
/// Simulation works by advancing the sim to time step t_n+1, and then iterating over every object in the sim to 
/// advance it to the next time step. Some of these objects will need to have a variable time step to accomodate 
/// discrete control interfaces and electronics.
pub struct Sim {
    objects: Vec<Box<dyn Simulatable>>,
    start_time: f64,
    current_time: f64,
    end_time: f64,
    dt: f64,
    pub steps: u64,
    is_gui: bool,
    datacom_port: String,
    frame_counter_update: u64,
    environments: Vec<Box<dyn environments::EnviromentalEffect>>,
}

impl Sim {

    /// Creates a new blank sim.
    pub fn new(dt: f64) -> Sim{
        Sim { 
            objects: vec![], 
            start_time: 0.0, 
            current_time: 0.0, 
            end_time: 0.0, 
            dt: dt, 
            steps: 0,
            ..Default::default()}
    }

    pub fn update(&mut self) {
        todo!();
    }

    pub fn set_end_time(&mut self, end_time: f64) {
        self.end_time = end_time;
    }
    /// Returns a reference to object with given ID. No error handling yet
    pub fn get_object(&mut self, id: usize) -> &mut Box<dyn Simulatable> {
        &mut self.objects[id]
    }

    /// Exports current sim state to JSON. Unimplemented
    pub fn export_current_state_to_json_str(&mut self) -> &str {
        todo!();
    }

    /// Runs sum until given time.
    pub fn run_until(&mut self, end_time: f64) -> Result<(), Error>{
        info!("Running sim until t={} seconds.", end_time);
        self.end_time = end_time;
        // let mut bar = ProgressBar::new((end_time / self.dt) as u64);
        let mut graphical_frame_counter: u64 = 0;
        let graphical_update_max = 5000;
        // let next_update = Duration::from_micros(33_333);
        let next_update = Duration::from_millis(100);
        let mut last_update_time = Instant::now();

        debug!("Is graphical? {}", self.is_gui);

        // Initialize random other vectors
        for object in &mut self.objects {
            object.observe_position_and_rotation();
            // debug!("{} state (post initialization): {}", object.get_name(), object.get_state());
        }

        

        while self.current_time < self.end_time {
            for object in &mut self.objects{
                // Skip if static
                match &object.get_physics_type() {
                    PhysicsType::Static => continue,
                    _ => {
                        // Advance to next time step t_n+1, and iterate over each object to 
                        let t = self.current_time;
                        object.integrate_to_t(t, &self.environments);
                        object.observe_position_and_rotation();
                        // debug!("{} state: {}", object.get_name(), object.get_state());
                        // println!("{}", self.get_object(i).get_state());
                    }
                }
            }


            // If it's a graphical step, generate an update packet and send it over network.
            // Only try this if the sim is supposed to be graphical.

            if self.is_gui {
                let time_check = Instant::now();
                if time_check.duration_since(last_update_time) >= next_update
                || graphical_frame_counter >= graphical_update_max {
                    // debug!("SENDING PACKET");
                    let packet = self.datacom_update_packet();
                    // std::thread::sleep(Duration::from_millis(500));
                    self.datacom_send_packet("10.0.0.107:8081", &packet)?;
                    graphical_frame_counter = 0;
                    last_update_time = Instant::now();
                }
                else {
                    graphical_frame_counter += 1;
                }
            }
            self.current_time = self.current_time + self.dt;
            self.steps += 1;
            // bar.inc(1);
        }

        Ok(())
    }

    pub fn run(&mut self) {
        let t_start = std::time::Instant::now();
        let _ = self.run_until(self.end_time);
        // Send final update packet
        if self.is_gui{
            let packet = self.datacom_update_packet();
            debug!("Final Packet: {}", packet);
            self.datacom_send_packet("10.0.0.107:8081", &packet).unwrap();
        }
        info!("Program finished in {:.2?} in {:} steps", t_start.elapsed(), self.steps);
    }

    /// Adds an object to the simulation.
    pub fn add_object(&mut self, object: Box<dyn Simulatable>) {
        self.objects.push(object);
    }

    /// Creates JSON for DATACOM initialization. Legacy/
    pub fn scene_initialization_to_datacom(&self) -> String {
        
        let mut json_str: String = "".to_string();
        
        for object in self.objects.iter() {
            json_str.push_str(object.json_object_initialization().to_string().as_str());
        }
        
        return json_str;
    }

    /// Exports all object state history to file.
    pub fn record_run(&mut self, folderpath: &str) {
        for object in &mut self.objects {
            let filepath = "".to_owned() + folderpath+"/object_"+object.get_name()+".csv";
            object.export_data(&filepath);
            object.combine_csv_files().unwrap();
        }
    }

    /// Loads and runs all scenarios for a given job
    pub fn load_and_run_job(folderpath: &str) -> Result<(), Error> {

        // Obtain job from the todo folder by fetching directories, and taking the first one in 
        // alphabetical order, then retrieve vector of scenarios
        let job = (get_directories("data/todo").unwrap())[0].to_string();
        debug!("JOB: {}", &job);
        let scenario_file_path = job.clone() + "/scenarios";
        let scenarios = get_files(&scenario_file_path).expect("No scenarios in file");

        // let mut scenario_vector = vec![];
        // for scenario in scenarios {
        //     let scenario_loaded = ScenarioInfo::new(job.clone(), scenario);
        //     debug!("JOB NAME: {}", scenario_loaded.job_name);
        //     debug!("SCENARIO NAME: {}", scenario_loaded.scenario_name);
        //     scenario_vector.push(scenario_loaded);
        // }

        // for scenario in scenario_vector {
        //     let mut sim = Sim::load_scenario(scenario);
        //     // sim.datacom_start("localhost:8080").expect("No connection established.");
        //     sim.run();
        //     sim.record_run("data/todo/default_name/output");
        // }

        // AI code
        let scenario_vector: Vec<_> = scenarios
            .par_iter() // Parallel iterator
            .map(|scenario| {
                let scenario_loaded = ScenarioInfo::new(job.clone(), scenario.clone());
                debug!("JOB NAME: {}", scenario_loaded.job_name);
                debug!("SCENARIO NAME: {}", scenario_loaded.scenario_name);
                scenario_loaded
            })
            .collect();

        let mut bar = RwLock::new(ProgressBar::new((scenario_vector.len()) as u64));

        scenario_vector.par_iter().for_each(|scenario| {
            let mut sim = Sim::load_scenario(scenario.clone());
            // sim.datacom_start("localhost:8080").expect("No connection established.");
            sim.run();
            sim.record_run("data/todo/default_name/output");
            bar.write().unwrap().inc(1);
        });

        

        

        Ok(())
    }

    /// Loads scenario from file. 
    pub fn load_scenario(scenario: ScenarioInfo) -> Self {
        let mut filepath = &scenario.scenario_name;
        // filepath.push_str("/scenario_setup.json");
        info!("Loading {}...", filepath);
        let json_unparsed = std::fs::read_to_string(filepath).unwrap();
        let mut json_parsed: Value = serde_json::from_str(&json_unparsed).unwrap();
        
        // Assign basic values
        let name = json_parsed["scenarioName"].as_str().unwrap();
        let end_time = json_parsed["endTime"].as_f64().unwrap();
        let dt = json_parsed["dtMin"].as_f64().unwrap();
        let mut Sim = Sim::new(dt);
        Sim.set_end_time(end_time);
        Sim.clear_environments(); // Remove me when default option is gone.


        // Assign port if one is in the file
        let datacom_port: String = match &json_parsed["datacomPort"] {
            // Value::Null => {"".to_string()},
            Value::String(port_name) => {port_name.to_string()},
            _ => {"".to_string()}
        };

        // Iterate over every vehicle with the worst pattern match ever made.
        // I can't figure out how to make this generic because I can't pass
        // `U` as a constant when read from a file.
        match &mut json_parsed["vehicles"] {
            Value::Array(data_vector) => {
                for mut obj in data_vector {
                    // debug!("{}", obj["input_size"]);
                    let U = obj["inputSize"].as_f64().unwrap() as usize;
                    let mut run_name = scenario.scenario_name.clone();
                    if let Some(stripped) = run_name.strip_suffix(".json") {
                        run_name = stripped.to_string();
                    }
                    debug!("Run name: {}", run_name);
                    obj["runName"] = Value::String(run_name);


                    let mut job_name = scenario.job_name.clone();
                    debug!("job name: {}", job_name);
                    obj["jobName"] = Value::String(job_name);

                    let vehicle: Box<dyn Simulatable> = match U {
                        1 => Box::new(Vehicle::<1>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        2 => Box::new(Vehicle::<2>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        3 => Box::new(Vehicle::<3>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        4 => Box::new(Vehicle::<4>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        5 => Box::new(Vehicle::<5>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        6 => Box::new(Vehicle::<6>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        7 => Box::new(Vehicle::<7>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        8 => Box::new(Vehicle::<8>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        9 => Box::new(Vehicle::<9>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        10 => Box::new(Vehicle::<10>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        11 => Box::new(Vehicle::<11>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        12 => Box::new(Vehicle::<12>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        13 => Box::new(Vehicle::<13>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        14 => Box::new(Vehicle::<14>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        15 => Box::new(Vehicle::<15>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        16 => Box::new(Vehicle::<16>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        17 => Box::new(Vehicle::<17>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        18 => Box::new(Vehicle::<18>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        19 => Box::new(Vehicle::<19>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        20 => Box::new(Vehicle::<20>::load_from_job(&(scenario.job_name.clone()), &mut obj).unwrap()),
                        _ => panic!("Unsupported number of inputs when creating vehicle: {}", U)
                    };
                    Sim.add_object(vehicle);
                    // debug!("Loaded object {}", obj["name"].as_str().unwrap());
                }
            }
            _ => {}
        }
        
        match &json_parsed["environments"] { 
            Value::Array(data_vector) => {
                for environment in data_vector {
                    let env_type = environment["type"].as_str().unwrap();
                    debug!("{}", env_type);
                    match env_type {
                        "PointMassGravity" => {
                            Sim.add_environment(Box::new(environments::GravitationalField::load_from_json_parsed(environment)))
                            },
                        "ConstantField" => {
                            // debug!("Constant Field");
                            Sim.add_environment(Box::new(environments::GravitationalField::load_from_json_parsed(environment)))
                        },
                        _ => {},
                    };
                }
            },
            _ => {}
        }

        // let mut Sim = Sim::new(dt);

        

        return Sim;
    }

    

    /// Adds data transmission port to DATACOM.
    pub fn add_datacom_port(&mut self, new_port: String) {
        self.datacom_port = new_port;
    }

    /// Creates JSON for DATACOM initialziation.
    pub fn initialize_datacom_json(&mut self) -> Value {
            let mut entities: Vec<serde_json::Value> = vec![];
            for object in &mut self.objects{
                debug!("Object name: {}", object.get_name());
                let temp = object.datacom_json_initialize();
                match &temp {
                    Some(Value) => entities.push(temp.expect("You shouldn't see this! ")),
                    _ => (),
                }
            }

            let datacom_packet = json!({
                "viewports": [],
                "entities": entities
            });

            // debug!("{}", datacom_packet.to_string());
            return datacom_packet;
    }

    /// Creates state update packet to send to DATACOM
    pub fn datacom_update_packet(&mut self) -> String {
            let mut commands: Vec<Value> = vec![];
            for i in 0..self.objects.len(){
                let temp = self.get_object(i).datacom_json_command_step();
                match &temp {
                    Some(Value) => {
                        let mut temp2 = temp.expect("You shouldn't see this");
                        commands.append(&mut temp2)},
                    _ => ()
                }
            }
            // debug!("{}", json!(commands).to_string());
            return json!(commands).to_string();
    }

    /// Sends packet to DATACOM
    pub fn datacom_send_packet(&self, addr: &str, data: &str) -> Result<(), Error> {

        let max_connection_attempts = 100;
        let mut num_attempts = 0;
        while max_connection_attempts > num_attempts {
            // debug!("Connection attempt {}", num_attempts+1);
            let connection_result = TcpStream::connect(addr);
            std::thread::sleep(std::time::Duration::from_millis(10));

            match &connection_result {
                Ok(_) => {
                    // debug!("Established connection. Transmitting data...");
                    let mut stream = connection_result.unwrap();
                    stream.write_all(&data.as_bytes()).unwrap();
                    // debug!("Successfully transmitted packet");
                    return Ok(());
                },
                Err(_) => {
                    // error!("Connection attempt failed");
                    num_attempts+=1;
                }
            }
        }

        error!("Connection attempt timed out.");
        Err(Error)
    }

    /// Transmits larger files in chunks to DATACOM
    pub fn datacom_send_large_file(&self, addr: &str, filepath: &str) -> Result<(), Error> {
        // debug!("Sending large file: {}", filepath);
        let mut packets_sent = 0;
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.write_all(filepath.as_bytes()).unwrap();
        std::thread::sleep(Duration::from_millis(500));
        let mut file = File::open(filepath).unwrap();
        let file_size = file.metadata().unwrap().len();
        // debug!("Filesize: {}", file_size);
        // std::thread::sleep(Duration::from_millis(500));
        // Send file size first
        stream.write_all(&file_size.to_be_bytes()).unwrap();
        
        let mut buffer = [0; CHUNK_SIZE];
        let mut bytes_sent = 0;
    
        while bytes_sent < file_size {
            // debug!("Sending packet {}", packets_sent+1);
            let bytes_read = file.read(&mut buffer).unwrap();
            if bytes_read == 0 {
                break;
            }
            stream.write_all(&buffer[..bytes_read]).unwrap();
            bytes_sent += bytes_read as u64;
            packets_sent += 1;
            // std::thread::sleep(Duration::from_millis(100));
        }
        Ok(())
    }

    /// Single function that initializes and sends start information to DATACOM link
    pub fn datacom_start(&mut self, datacom_addr: &str) -> Result<(), Error> {
        // Set relevant data flags
        self.is_gui = true;
        self.add_datacom_port(self.datacom_port.to_string());

        // Generate initial data JSON and attempt to connect to DATACOM
        let initialization_packet = self.initialize_datacom_json();
        self.datacom_send_packet(datacom_addr, &initialization_packet.to_string())?;
        debug!("Inititialization Packet: {}", initialization_packet);
        info!("Initialization packet transmitted.");
        // Retrieve and send model files      
        // std::thread::sleep(Duration::from_millis(1000));
        let unique_model_names = self.get_unique_model_names()?;
        for i in unique_model_names.into_iter() {
            info!("Transmitting packet for {}", &i);
            // let file = std::fs::read_to_string(i).expect("Failed to read file!");
            // debug!("File:{}", file.as_str());
            // self.datacom_send_packet(datacom_addr, file.as_str())?;
            self.datacom_send_large_file(datacom_addr, &i).unwrap();
        }
        self.datacom_send_packet(datacom_addr, "END")?;
        info!("Models transmitted.");


        Ok(())
    }

    /// Iterates over objects and gets unique model names
    pub fn get_unique_model_names(&mut self) -> Result<Vec<String>, Error> {
        let mut unique_model_paths: Vec<String> = vec![];
        for i in 0..self.objects.len() {
            let model_name = &self.get_object(i).get_model_path();
            
            if unique_model_paths.iter().any(|e| model_name.contains(e)) && model_name == &"" {
                debug!("{} already in model names", model_name);
            }
            else {
                unique_model_paths.push(model_name.to_string());
            }
        }

        Ok(unique_model_paths)
    }

    pub fn add_environment(&mut self, new_environment: Box<dyn environments::EnviromentalEffect>) {
        self.environments.push(new_environment);
    }

    pub fn clear_environments(&mut self) {
        self.environments = vec![];
    }
}

impl Default for Sim {
    fn default() -> Self {
        Sim {
            objects: vec![], 
            start_time: 0.0, 
            current_time: 0.0, 
            end_time: 0.0, 
            dt: 1.0E-3, 
            steps: 0,
            is_gui: false,
            datacom_port: "".to_string(),
            frame_counter_update: 10,
            environments: vec![
                Box::new(
                    environments::GravitationalField::default()
                )
            ]
        }
    }
    
}

/// Behavior recquired for any object to be present in the sim
pub trait Simulatable {
    // Getter and Setter functions
    fn get_name(&self) -> &str;
    fn set_name(&mut self, new_name: &str);
    fn get_position(&self) -> na::Point3<f64>;
    fn set_position(&mut self, new_position: na::Point3<f64>);
    fn get_rotation(&self) -> na::Point3<f64>;
    fn set_rotation(&mut self, new_rotation: na::Point3<f64>);
    fn get_state(&self) -> SMatrix<f64, 12, 1>;
    fn set_state(&mut self, new_x: State);
    fn get_physics_type(&self) -> &PhysicsType;
    fn set_physics_type(&mut self, new_type: PhysicsType);
    fn get_model_path(&self) -> &str;

    fn integrate_to_t(&mut self, t: f64, environment_vector: &Vec<Box<dyn environments::EnviromentalEffect>>);
    fn observe_position_and_rotation(&mut self);
    
    fn get_xdot(&self, x: &State, t: &f64, environment_vector: &Vec<Box<dyn environments::EnviromentalEffect>>) -> State;
    
    fn get_component_forces(&mut self, t: f64) -> State;
    fn json_object_initialization(&self) -> sj::Value;
    fn status_to_json(&self) -> sj::Value;
    fn export_data(&mut self, filepath: &str);
    fn add_component(&mut self, new_component: Box<dyn ComponentPart>);
    fn datacom_json_initialize(&mut self) -> Option<serde_json::Value>;
    fn datacom_json_command_step(&self) -> Option<Vec<Value>>;
    fn calculate_environmental_acceleration(&self, environment_vector: &Vec<Box<dyn environments::EnviromentalEffect>>) -> State;
    fn combine_csv_files(&self) -> std::io::Result<()>;
}

/// Atomic counter for enforcing unique object IDs
static OBJECT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Struct for dynamic controllable physics objects in the sim. Has an integrator, state space matricies, and flight computer.
pub struct Vehicle<const U: usize> {
    name: String,
    id: u64,
    mass: f64,
    A: na::SMatrix<f64, 12, 12>,
    B: na::SMatrix<f64, 12, U>,
    C: na::SMatrix<f64, 6, 12>,
    x: State,
    u: na::SMatrix<f64, U, 1>,
    fc: Box<dyn FlightControl<U>>,
    position: na::Point3<f64>,
    rotation: na::Point3<f64>,
    last_time: f64,
    motors: Vec<Box<dyn ComponentPart>>,
    integrator: Integrator,
    data: DataLogger<12, U>,
    is_graphical: bool,
    graphical_info: graphical::GraphicalData,
    physics_type: PhysicsType,
}


impl<const U: usize> Vehicle<U> {
    pub fn new() -> Self {
        Vehicle{ name: "".to_string(), 
        id: OBJECT_COUNTER.fetch_add(1, Ordering::Relaxed), 
        mass: 1.0, // kg
        A: na::SMatrix::zeros(), 
        B: na::SMatrix::zeros(),
        C: na::SMatrix::zeros(),
        x: na::SMatrix::zeros(),
        u: na::SMatrix::zeros(),
        fc: Box::new(NullComputer::<U>::new()), 
        position: na::Point3::origin(), 
        rotation: na::Point3::origin(), 
        last_time: 0.0,
        motors: vec![],
        integrator: Integrator::RK4,
        data: DataLogger::<12, U>::new(5.0, 0, &""),
        is_graphical: true,
        ..Default::default()
     }
    }

    pub fn set_mass(&mut self, new_mass:f64) {
        self.mass = new_mass;
    }

    pub fn get_xdot_A(&self) -> State {
        return self.A*self.x;
    }

    pub fn get_xdot_B(&self, u: na::SMatrix<f64, U, 1>) -> SMatrix<f64, 12, 1> {
        return self.B*u;
    }

    pub fn get_u(&self, current_state: State) -> na::SMatrix<f64, U, 1> {

        let mut u = Inputs::<U>::zeros();
        for (i, motor) in self.motors.iter().enumerate() {
            u[i] = motor.get_force();
        }

        return u
    }

    pub fn calculate_and_update_u(&mut self, t: f64, current_state: State) {
        let u_cmd = self.fc.calculate_u(t, current_state);
        let mut i=0;
        for motor in &mut self.motors {
            motor.set_force(u_cmd[i]);
            motor.update_to_time(vec![0.0], t);
            i += 1;
        }
    }


    pub fn get_A(&self) -> &na::SMatrix<f64, 12, 12>{
        &self.A
    }

    pub fn set_A(&mut self, A_new: na::SMatrix<f64, 12, 12>) {
        self.A = A_new;
    }

    pub fn get_B(&self) -> &na::SMatrix<f64, 12, U>{
        &self.B
    }

    pub fn set_B(&mut self, B_new: na::SMatrix<f64, 12, U>) {
        self.B = B_new;
    }

    pub fn get_C(&self) -> &na::SMatrix<f64, 6, 12> {
        &self.C
    }

    pub fn set_C(&mut self, C_new: na::SMatrix<f64, 6, 12>) {
        self.C = C_new;
    }

    /// Loads vehicle using dual path data storage for type and instance in sim
    pub fn load_from_job(job_name: &str, instance_info: &mut sj::Value) -> Result<Vehicle<U>, std::io::Error> {
        let filepath = format!("{}/objects/{}/{}.json", job_name, instance_info["name"].as_str().unwrap(), instance_info["name"].as_str().unwrap());
        debug!("Object Loaded Filepath: {}", filepath);
        let json_unparsed = std::fs::read_to_string(filepath).unwrap();
        let mut json_parsed: Value = serde_json::from_str(&json_unparsed).unwrap();
        
        json_parsed["state"] = instance_info["state"].take();
        json_parsed["name"] = instance_info["name"].take();
        json_parsed["physicsType"] = instance_info["physicsType"].take();
        json_parsed["runName"] = instance_info["runName"].take();
        json_parsed["jobName"] = instance_info["jobName"].take();

        Ok(Vehicle::load_from_json_parsed(&json_parsed))
    }

    /// Load vehicle from a JSON file
    pub fn load_from_json(filepath: &str) -> Vehicle<U> {
        todo!();
    }

    /// Loads vehicle from a JSON in string format
    pub fn load_from_json_string(json_unparsed: &str) -> Vehicle<U> {
        todo!();
    }

    /// Loads vehicle from a full JSON object
    pub fn load_from_json_parsed(json_parsed: &sj::Value) -> Vehicle<U> {
        let name = json_parsed["name"].as_str().unwrap();
        let mass = json_parsed["mass"].as_f64().unwrap();
        let id = OBJECT_COUNTER.fetch_add(1, Ordering::Relaxed);

        let A_dat: Vec<f64> = json_parsed["A"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|x| x.as_f64().unwrap())
            .collect();

        let A: na::SMatrix<f64, 12, 12> = na::SMatrix::from_row_slice(&A_dat[..]);

        let B_dat: Vec<f64> = json_parsed["B"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|x| x.as_f64().unwrap())
            .collect();

        debug!("B SIZE: {}", B_dat.len());

        let B: na::SMatrix<f64, 12, U> = na::SMatrix::from_row_slice(&B_dat[..]);

        let x_dat: Vec<f64> = json_parsed["state"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|x| x.as_f64().unwrap())
            .collect();

        let x: na::SMatrix<f64, 12, 1> = na::SMatrix::from_row_slice(&x_dat[..]);

        let (physics_type, integrator) = match json_parsed["physicsType"].as_str().unwrap() {
            "RK4" => {
                (PhysicsType::StateSpace, Integrator::RK4)
            },
            "ForwardEuler" => {
                (PhysicsType::StateSpace, Integrator::ForwardEuler)
            }
            _ => {
                (PhysicsType::Static, Integrator::RK4)
            },
        };

        debug!("Name: {}", name);
        debug!("ID: {}", id);
        // debug!("ID: {}", id);
        // debug!("ID: {}", id);
        debug!("Mass: {}", mass);
        // debug!("A: {}", A);
        // debug!("B: {}", B);
        // debug!("x: {}", x);
        // debug!("Integrator: {}", integrator);

        let sample_time: f64 = match &json_parsed["SampleTime"] {
            Value::Number(t_s) => {t_s.as_f64().unwrap()},
            _ => {0.0}
        };

        let max_steps: u64 = match &json_parsed["MaxSteps"] {
            Value::Number(t_s) => {t_s.as_u64().unwrap()},
            _ => {0}
        };

        let mut run_name: &str = match &json_parsed["runName"] {
            Value::String(run_name) => run_name,
            _=> {
                warn!("No run name found!");
                ""
            },
        };

        let mut job_name: &str = match &json_parsed["jobName"] {
            Value::String(run_name) => run_name,
            _=> {warn!("No job name found!");
                ""},
        };

        let prefix = job_name.to_string() + "/scenarios/";
        if let Some(stripped) = run_name.strip_prefix(&prefix) {
            run_name = stripped;
        }

        debug!("Stripped run name: {}", run_name);
        // let scenario = run_name.strip_prefix(job_name);
        let storage_directory = format!("{}/output/{}", job_name, run_name);
        fs::create_dir_all(storage_directory).expect("Could not create output directory.");
        let storage_directory = format!("{}/output/{}/object_{}_{}", job_name, run_name, id, name);
        debug!("Storage Directory: {}", storage_directory);

        let mut data_recorder = DataLogger::<12,U>::new(sample_time, max_steps as usize, &storage_directory);

        let mut components: Vec<Box<dyn ComponentPart>> = vec![];
        match &json_parsed["components"] {
            Value::Array(data_vector) => {
                for obj in data_vector {
                    // debug!("Component String: {}", obj);
                    let component_type = obj["component_type"].as_str().unwrap();
                    debug!("Component Type: {}", component_type);
                    match component_type {
                        "IdealThruster" => {
                            components.push(Box::new(IdealThruster::load_from_json_parsed(obj.clone())));
                        },
                        _ => {}
                    }
                }
            }
            _ => {}
        }

        let fc: Box<dyn fc::FlightControl<U>> = match &json_parsed["flight_controller"] {
            
            Value::Object(obj) => {
                debug!("{}", obj["fcType"].as_str().unwrap());
                match obj["fcType"].as_str().unwrap() {
                    "StateFeedback" => {
                        Box::new(FlightComputer::new_from_json(&json_parsed["flight_controller"]))
                    },
                    "NeuralNet" => {
                        Box::new(NNComputer::new_from_json(&json_parsed["flight_controller"]).unwrap())
                    }
                    _ => {
                        warn!("Invalid flight controller type specified!");
                        Box::new(NullComputer::new())
                    }
                }
            },
            _ => {
                warn!("NO FLIGHT CONTROLLER FOUND");
                Box::new(NullComputer::new())
            },
        };

        // let fc: Box<dyn fc::FlightControl<U>> = match &json_parsed["flight_controller"] { 
        //     Value::String(fc_type) => {
        //         warn!("NO FLIGHT CONTROLLER FOUND");
        //         Box::new(NullComputer::new())
        //     }
        //     _=> {
        //         Box::new(FlightComputer::new_from_json(&json_parsed["flight_controller"]))
        //     }
        // };

        


        // debug!("A: {}", A);

        let mut vehicle = Vehicle::<U> {
            name: name.to_string(),
            id: id,
            mass: mass,
            // id: id,
            A: A,
            B: B,
            C: na::SMatrix::zeros(),
            x: x,
            u: na::SMatrix::zeros(),
            fc: fc,
            position: na::Point3::<f64>::origin(),
            rotation: na::Point3::<f64>::origin(),
            last_time: -1.0,
            motors: components,
            integrator: integrator,
            physics_type: physics_type,
            is_graphical: false,
            graphical_info: graphical::GraphicalData{..Default::default()},
            data: data_recorder,
            // ..Default::default()
        };

        match &json_parsed["GraphicalElements"] {
            Value::Array(data_vector) => {
                for graph in data_vector{
                    vehicle.set_model(graphical::GraphicalData::from_json(&graph));
                }
            },
            _=> {
                // graphical::GraphicalData::new("", "data/test_object/test.obj", [1.0, 1.0, 1.0, 1.0])
            }
        };


        


        return vehicle;
        // todo!();
    }

    pub fn add_flight_controller(&mut self, new_fc: Box<dyn FlightControl<U>>) {
        self.fc = new_fc;
    }

    pub fn datacom_position_command(&self) -> Value {
        let pos_cmd = json!({
            "targetEntityID": self.id,
            "commandType": "EntityChangePosition",
            "data": get_point_as_list(self.position*GRAPHICAL_SCALING_FACTOR),
        });
        return pos_cmd;
    }

    pub fn datacom_rotation_command(&self) -> Value {
        let rot_cmd = json!({
            "targetEntityID": self.id,
            "commandType": "EntityRotate",
            "data": get_point_as_list(self.rotation),
        });

        return rot_cmd
    }

    pub fn get_component(&mut self, id: usize) -> &mut Box<dyn ComponentPart> {
        &mut self.motors[id]
    }

    pub fn set_model(&mut self, new_graphical_info: GraphicalData) {
        self.is_graphical=true;
        self.graphical_info = new_graphical_info;
    }
    
    pub fn set_id(&mut self, new_id: u64) {
        self.id = new_id;
    }

    pub fn get_cmd_position(&self) -> State {
        self.fc.get_cmd_position()
    }
}

impl<const U: usize> Simulatable for Vehicle<U> {

    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn set_name(&mut self, new_name: &str) {
        self.name = new_name.to_string();
    }

    fn integrate_to_t(&mut self, t: f64, environment_vector: &Vec<Box<dyn environments::EnviromentalEffect>>) {
        let dt = t - self.last_time;
        self.calculate_and_update_u(t, self.x);
        self.x = self.integrator.integrate(
            |x, t| self.get_xdot(&x, &t, &environment_vector), &self.x, &self.last_time, &t);

        // debug!("Input Vector at t={}: {}", t, self.get_u(self.x));
        // debug!("State Vector at t={}: {}", t, self.x);
        
        match self.physics_type {
            PhysicsType::Static => {
                if self.last_time > 0.0 {
                    self.data.record(t, self.get_state(), self.get_cmd_position(), self.get_u(self.x));
                }
            }
            _ => {
                self.data.record(t, self.get_state(), self.get_cmd_position(), self.get_u(self.x));
            }
        }

        self.last_time = t;        
    }

    /// Calculates x_dot using the state space equation: \dot{x} = A\vec{x} + B\vec{u}
    fn get_xdot(&self, x: &State, t: &f64, environment_vector: &Vec<Box<dyn environments::EnviromentalEffect>>) -> State {
        let x_dot_env = self.calculate_environmental_acceleration(environment_vector); 
        // debug!("B: {}", self.B);
        // debug!("u: {}", self.get_u(*x));
        // debug!("x: {}", self.x);
        // debug!("B*u: {}", self.B*self.get_u(*x));
        // debug!("xdot_env: {}", x_dot_env);
        self.A*x
            + rotation_frame(&x[6], &x[7], &x[8])*self.B*self.get_u(*x)
            + x_dot_env
    }

    fn observe_position_and_rotation(&mut self) {
        self.position = na::Point3::new(
            self.x[0], 
            self.x[1], 
            self.x[2]
        );

        self.rotation = na::Point3::new(self.x[6], self.x[7], self.x[8]);
    }

    fn get_position(&self) -> na::Point3<f64> {
        self.position
    }
    
    fn set_position(&mut self, new_position: na::Point3<f64>) {
        self.position = new_position;
    }
    
    fn get_rotation(&self) -> na::Point3<f64> {
        self.rotation
    }

    fn set_rotation(&mut self, new_rotation: na::Point3<f64>) {
        todo!();
    }

    fn get_state(&self) -> SMatrix<f64, 12, 1> {
        self.x
    }

    fn set_state(&mut self, new_x: State) {
        self.x = new_x;
    }

    fn get_component_forces(&mut self, t: f64) -> State {
        let mut component_force_vec = na::SMatrix::<f64, 12, 1>::zeros();
        for component in &mut self.motors {
            component.update_to_time(vec![], t);
            component_force_vec = component_force_vec + component.get_force_on_parent();
        };
        debug!("Component Force Vector at t={}: {}", t, component_force_vec);
        return component_force_vec;
    }

    fn json_object_initialization(&self) -> sj::Value {
        let json_str = format!("{{
            \"Id\": {:.?},
            \"Name\": \"{}\",
            \"Position\": [{:.?}, {:.?}, {:.?}],
            \"Rotation\": [{:.?}, {:.?}, {:.?}],
            \"Scale\": [1.0, 1.0, 1.0]
            \"obj_file\": \"{}\"
        }}",
        self.id,
        self.name,
        self.position[0],
        self.position[1],
        self.position[2],
        self.rotation[0],
        self.rotation[1],
        self.rotation[2],
        "Default");

        // debug!("{}", json_str);

        let json = sj::from_str(&json_str).unwrap();
        return json;
    }

    fn status_to_json(&self) -> sj::Value {
        let json_str = format!("
            \"id\": {},
            \"name\": {},
            \"position\": [{:.?}, {:.?}, {:.?}],
            \"rotation\": [{:.?}, {:.?}, {:.?}],
        ",
        self.id,
        self.name,
        self.position[0],
        self.position[1],
        self.position[2],
        self.rotation[0],
        self.rotation[1],
        self.rotation[2],
        );

        let json = sj::from_str(&json_str).unwrap();
        return json;

    }

    fn export_data(&mut self, filepath: &str) {
        self.data.to_csv(filepath);
    }

    fn add_component(&mut self, new_component: Box<dyn ComponentPart>) {
        self.motors.push(new_component);
    }

    fn datacom_json_initialize(&mut self) -> Option<sj::Value> {
        if self.is_graphical {
            // Generate list of models to put into 
            let mut model_list: Vec<Value> = vec![];
            let top_level_model = self.graphical_info.get_model_value();
            model_list.push(top_level_model);
            // for i in 0..self.motors.len() {
            //     let temp = self.get_component(i).datacom_get_model_json();
            //     model_list.push(temp);
            // }
            
            let json_file = Option::Some(json!({
                "Name": self.name,
                "id": self.id,
                "Scale": self.graphical_info.scale,
                "Position": [self.position[0]*GRAPHICAL_SCALING_FACTOR, self.position[1]*GRAPHICAL_SCALING_FACTOR, self.position[2]*GRAPHICAL_SCALING_FACTOR],
                "Rotation": [self.rotation[0], self.rotation[1], self.rotation[2]],
                "Models": model_list,
            }));
            return json_file
            
        }
        
        else {
            return Option::None;
        };

        // let json: Value = 
    }

    // fn add_flight_controller(&self, new_fc: Box<dyn FlightControl<U>>) {

    // }

    fn datacom_json_command_step(&self) -> Option<Vec<Value>> {
        let mut cmd_vec = vec![];
        cmd_vec.push(self.datacom_position_command());
        cmd_vec.push(self.datacom_rotation_command());
        // Add behavior to push subcomponent commands here as well

        return Option::Some(cmd_vec);
    }

    /// Getter method for model path
    fn get_model_path(&self) -> &str {
        &self.graphical_info.get_model_path()
    }

    /// Getter method for vehicle physics type.
    fn get_physics_type(&self) -> &PhysicsType {
        &self.physics_type
    }

    /// Setter for vehicle physics type
    fn set_physics_type(&mut self, new_type: PhysicsType) {
        self.physics_type = new_type;
    }

    fn calculate_environmental_acceleration(&self, environment_vector: &Vec<Box<dyn environments::EnviromentalEffect>>) -> State {
        
        let environment_state_xdot: State =  environment_vector.into_iter().map(|environment| environment.calculate_acceleration_on_object(&self.mass, &self.position)).sum();
        return environment_state_xdot;
    }

    fn combine_csv_files(&self) -> std::io::Result<()> {
        self.data.combine_csv_files()
    }
}

impl<const U: usize> Default for Vehicle<U> {
    fn default() -> Self {
        Vehicle {
            name: "DEFAULT".to_string(),
            id: OBJECT_COUNTER.fetch_add(1, Ordering::Relaxed),
            mass: 0.0,
            A: na::SMatrix::zeros(),
            B: na::SMatrix::zeros(),
            C: na::SMatrix::zeros(),
            x: na::SMatrix::zeros(),
            u: na::SMatrix::zeros(),
            fc: Box::new(NullComputer::new()),
            position: na::Point3::<f64>::origin(),
            rotation: na::Point3::<f64>::origin(),
            last_time: -1.0,
            motors: vec![],
            integrator: Integrator::ForwardEuler,
            data: DataLogger::<12, U>::new(0.0, 0, "DEFAULT"),
            is_graphical: true,
            // obj_file: "DEFAULT".to_string(),
            graphical_info: graphical::GraphicalData{..Default::default()},
            physics_type: PhysicsType::StateSpace,
        }
    }
}

// UTILITIES

/// Physics type classifier for simulated objects
pub enum PhysicsType {
    Static,
    StaticTrajectory,   // Currently unused
    StateSpace,
}

/// Activation enum. Can render a component active or inactive.
pub enum Status {
    Active,
    Inactive
}

impl Status {
    pub fn activate() {
        todo!()
    }
}

/// Enum based integrator option.
pub enum Integrator {
    ForwardEuler,
    RK4,
}

impl Integrator {
    /// Time step integration which accepts a function to get state vector derivative
    pub fn integrate<F, const S: usize>(&self, f: F, y_old: &na::SMatrix<f64, S, 1>, t_last: &f64, t_new: &f64) -> SMatrix<f64, S, 1> 
    where 
        F: Fn(&SMatrix<f64, S, 1>, &f64) -> SMatrix<f64, S, 1>
    {

        let y_new: na::SMatrix<f64, S, 1> = match self {
            Integrator::ForwardEuler => {
                let h = (t_new-t_last);
                let y_prime = f(y_old, t_new);
                y_old + y_prime * h
            },
            Integrator::RK4 => {
                let h = t_new-t_last;
                let k1 = f(y_old, t_last);
                let k2 = f(&(y_old+k1*h/2.0), &(t_last+h/2.0));
                let k3 = f(&(y_old+k2*h/2.0),&(t_last+h/2.0));
                let k4 = f(&(y_old+h*k3), &(t_last+h));

                y_old + h/6.0*(k1+2.0*k2+2.0*k3+k4)
                },
            _ => todo!(),
        };

        return y_new;
    }
}

/// Function to return a state vector with a negative z component
fn z_down() -> State {
    na::SMatrix::from_row_slice(&[
        0.0,        // x
        0.0,        // y
        -1.0,       // z
        0.0,        // xdot
        0.0,        // ydot
        0.0,        // zdot
        0.0,        // phi
        0.0,        // theta
        0.0,        // psi
        0.0,        // phidot
        0.0,        // thetadot
        0.0         // psidot
    ])
}

/// Function to return a state vector with a negative z acceleration component
fn z_acc_down() -> State {
    na::SMatrix::from_row_slice(&[
        0.0,        // xdot
        0.0,        // ydot
        0.0,       // zdot
        0.0,        // xdotdot
        0.0,        // ydotdot
        -1.0,        // zdotdot
        0.0,        // phidot
        0.0,        // thetadot
        0.0,        // psidot
        0.0,        // phidotdot
        0.0,        // thetadotdot
        0.0         // psidotdot
    ])
}

// fn forward_matrix(dt: f64) -> SMatrix<f64, 12, 12> {
//     na::SMatrix::<f64, 12, 12>::from_row_slice(&[
//         0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
//         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
//         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 

//         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
//         0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
//         0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 

//         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
//         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
//         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 

//         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
//         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
//         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
//     ])
// }


// #####################


pub struct DataLogger<const S: usize, const U: usize> {
    time_vector: Vec<f64>,
    state_vector: Vec<na::SMatrix<f64, S, 1>>,
    cmd_vector: Vec<State>,
    input_vector: Vec<Inputs<U>>,
    sample_time: f64, 
    last_time: f64,
    max_steps: usize,
    storage_directory: String,
    num_refresh: u64,
}

impl<const S: usize, const U:usize> DataLogger<S, U> {

    /// Create new datalogger
    pub fn new(sample_time: f64, max_steps: usize, storage_directory: &str) -> Self {

        // debug!("Sample time: {}", sample_time);
        let mut d = DataLogger{
            time_vector: vec![],
            state_vector: vec![],
            cmd_vector: vec![],
            input_vector: vec![],
            sample_time: sample_time,
            last_time: -100.0,
            max_steps: max_steps,
            storage_directory: storage_directory.to_string(),
            num_refresh: 0
        };

        clean_parent_directory(storage_directory);

        debug!("Datalogger filepath: {}", d.storage_directory);
        return d;
    }

    /// Function to record data, with checks for sample time.
    pub fn record(&mut self, time: f64, state: na::SMatrix<f64, S, 1>, cmd_state: State, input: Inputs<U>) {
        
        // Log data if sample time indicates it is ready, or bypass if values are null. 
        if ((time - self.last_time) >= self.sample_time) 
        || self.sample_time==0.0 
        || self.how_many_steps()<1 {
            self.time_vector.push(time);
            self.state_vector.push(state);
            self.cmd_vector.push(cmd_state);
            self.input_vector.push(input);
            self.last_time = time;
            // debug!("Last Command: {}", self.cmd_vector[self.cmd_vector.len()-1]);
            
            // Push data to file if it exceeds the maximum steps parameter
            // Only needs to be done if recording a step. Purge afterwards
            if self.time_vector.len() >= self.max_steps 
            && self.max_steps > 0{
                let filepath = format!("{}_{}.csv", self.storage_directory, self.num_refresh);
                self.to_csv(&filepath);
            }
        }

    }

    /// Exports datalogger items to file
    pub fn to_csv(&mut self, filepath: &str) {
        // Setup
        let filepath = format!("{}_{}.csv", self.storage_directory, self.num_refresh);
        // debug!("to_csv filepath: {}", filepath);
        let mut csv: String = "Time,X Position,Y Position,Z Position,x_vel,y_vel,z_vel,Pitch,Roll,Yaw,pitch_vel,roll_vel,yaw_vel".to_string();
        csv = csv 
            + ",x_pos_cmd"
            + ",y_pos_cmd"
            + ",z_pos_cmd"
            + ",x_vel_cmd"
            + ",y_vel_cmd"
            + ",z_vel_cmd"
            + ",pitch_cmd"
            + ",roll_cmd"
            + ",yaw_cmd"
            + ",pitch_vel_cmd"
            + ",roll_vel_cmd"
            + ",yaw_vel_cmd";
        if U > 1 {
            for (i, item) in ((&self.input_vector[0]).iter().enumerate()) {
                csv.push_str(&format!(",U_{}", i));
            }
        }
        csv.push_str("\n");


        // Iterate over all datapoints
        for (i, _) in self.time_vector.iter().enumerate() {
            let mut local_str = "".to_string();
            // debug!("Command Vectors: {}", &self.state_vector[i]);
            local_str = local_str + &format!("{:?}", self.time_vector[i]); // Time
            for state_el in (&self.state_vector[i]).into_iter() {
                local_str.push_str(&format!(",{:?}", state_el));
            }
            for cmd_el in (&self.cmd_vector[i]).into_iter() {
                local_str.push_str(&format!(",{:?}", cmd_el));
            }
            // debug!("{}", &self.cmd_vector[i]);
            if U > 1 {
                for input in (&self.input_vector[i]).into_iter() {
                    local_str.push_str(&format!(",{:?}", input));
                }
            }
            
            local_str.push_str("\n");
            csv.push_str(&local_str);
        }

        // Write to file
        // debug!("{}", filepath);
        std::fs::write(filepath, csv).unwrap();

        // Clear data vectors, increase the number of refreshes. 
        self.time_vector = vec![];
        self.state_vector = vec![];
        self.input_vector = vec![];
        self.cmd_vector = vec![];
        self.num_refresh = self.num_refresh + 1;
 
    }

    pub fn how_many_steps(&self) -> u64 {
        self.time_vector.len() as u64
    }

    pub fn combine_csv_files(&self) -> std::io::Result<()> {
        use std::fs::{self, File};
        use std::io::{BufReader, BufRead, Write, BufWriter};
        use std::path::Path;
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;
        
        // Structure to hold a line of data with its timestamp
        #[derive(Debug)]
        struct TimeRow {
            timestamp: f64,
            line: String,
            file_index: usize,
        }

        // Custom ordering for TimeRow to create a min-heap
        impl Ord for TimeRow {
            fn cmp(&self, other: &Self) -> Ordering {
                other.timestamp.partial_cmp(&self.timestamp).unwrap_or(Ordering::Equal)
            }
        }

        impl PartialOrd for TimeRow {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl PartialEq for TimeRow {
            fn eq(&self, other: &Self) -> bool {
                self.timestamp == other.timestamp
            }
        }

        impl Eq for TimeRow {}

        let base_path = Path::new(&self.storage_directory);
        let combined_path = format!("{}.csv", self.storage_directory);
        
        // Get all CSV files that match our pattern
        let mut files: Vec<String> = Vec::new();
        for i in 0..=self.num_refresh {
            let filename = format!("{}_{}.csv", self.storage_directory, i);
            if Path::new(&filename).exists() {
                files.push(filename);
            }
        }

        if files.is_empty() {
            return Ok(());
        }

        // Create file readers for each input file
        let mut readers: Vec<BufReader<File>> = files
            .iter()
            .map(|f| BufReader::new(File::open(f).unwrap()))
            .collect();

        // Create the output file with a buffer
        let output_file = File::create(&combined_path)?;
        let mut writer = BufWriter::new(output_file);

        // Copy header from first file
        let mut header = String::new();
        readers[0].read_line(&mut header)?;
        writer.write_all(header.as_bytes())?;

        // Skip headers in other files
        for reader in readers.iter_mut().skip(1) {
            let mut temp = String::new();
            reader.read_line(&mut temp)?;
        }

        // Initialize priority queue for sorting
        let mut heap = BinaryHeap::new();
        let mut line_buffers: Vec<String> = vec![String::new(); readers.len()];
        
        // Read first line from each file
        for (i, reader) in readers.iter_mut().enumerate() {
            if reader.read_line(&mut line_buffers[i])? > 0 {
                if let Some(timestamp) = parse_timestamp(&line_buffers[i]) {
                    heap.push(TimeRow {
                        timestamp,
                        line: line_buffers[i].clone(),
                        file_index: i,
                    });
                }
            }
        }

        // Process lines in timestamp order
        while let Some(TimeRow { line, file_index, .. }) = heap.pop() {
            // Write the current line
            writer.write_all(line.as_bytes())?;
            
            // Read next line from the file we just used
            line_buffers[file_index].clear();
            if readers[file_index].read_line(&mut line_buffers[file_index])? > 0 {
                if let Some(timestamp) = parse_timestamp(&line_buffers[file_index]) {
                    heap.push(TimeRow {
                        timestamp,
                        line: line_buffers[file_index].clone(),
                        file_index,
                    });
                }
            }
        }

        // Ensure all data is written
        writer.flush()?;

        // Clean up component files
        for file_path in files {
            fs::remove_file(file_path)?;
        }

        Ok(())
    }

    
} 

impl<const S: usize, const U: usize> Default for DataLogger<S, U> {
    fn default() -> Self {
        DataLogger {
            time_vector: vec![],
            state_vector: vec![],
            cmd_vector: vec![],
            input_vector: vec![],
            sample_time: 0.0,
            last_time: -100.0,
            max_steps: 0,
            num_refresh: 0,
            storage_directory: "data/test/test".to_string(),
        }
    }
}

/// Function to transform nalgebra point to list of floats
fn get_point_as_list(point: na::Point3<f64>) -> [f64; 3] {
    return [point[0], point [1], point[2]];
}

// ################################################################################################

#[derive(Debug, Deserialize)]
pub struct NetworkInfo {
    addresses: Vec<String>,
    ports: Vec<u16>,
    timeout_seconds: Duration,
    last_valid_address: Option<String>,
    last_valid_port: Option<u16>
}

/// Fetches directories
fn get_directories<P: AsRef<Path>>(path: P) -> Result<Vec<String>, std::io::Error>{
    let entries = fs::read_dir(path)?;
    
    Ok(entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            
            if path.is_dir() {
                path.to_str().map(String::from)
            } else {
                None
            }
        })
        .collect())
}

/// Fetches files
fn get_files(path: &str) -> Result<Vec<String>, std::io::Error> {
    Ok(fs::read_dir(path)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .filter_map(|entry| entry.path().to_str().map(String::from))
        .collect())
}
fn clean_parent_directory<P: AsRef<Path>>(dir_path: P) -> io::Result<bool> {
    let dir_path = dir_path.as_ref();
    
    // Get parent directory
    let parent_dir = dir_path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "No parent directory exists (might be root)"
        )
    })?;
    
    // Check if parent directory exists and is a directory
    if !parent_dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Parent path is not a directory"
        ));
    }

    // Read directory entries
    let entries = fs::read_dir(parent_dir)?;
    let mut has_files = false;
    let mut files_to_delete = Vec::new();

    // First pass: check for files and collect them
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            has_files = true;
            files_to_delete.push(path);
        }
    }

    // If files were found, delete them
    if has_files {
        for file_path in files_to_delete {
            fs::remove_file(file_path)?;
        }
    }

    Ok(has_files)
}

// Keep the original function for flexibility
fn clean_directory<P: AsRef<Path>>(dir_path: P) -> io::Result<bool> {
    let dir_path = dir_path.as_ref();
    
    if !dir_path.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "Specified path is not a directory"
        ));
    }

    let entries = fs::read_dir(dir_path)?;
    let mut has_files = false;
    let mut files_to_delete = Vec::new();

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            has_files = true;
            files_to_delete.push(path);
        }
    }

    if has_files {
        for file_path in files_to_delete {
            fs::remove_file(file_path)?;
        }
    }

    Ok(has_files)
}

/// Helper function to parse timestamp from a CSV line
fn parse_timestamp(line: &str) -> Option<f64> {
    line.split(',')
        .next()
        .and_then(|s| s.trim().parse::<f64>().ok())
}