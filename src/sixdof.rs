use core::num;
use std::fmt::Error;
use std::u64;
use std::{fmt::format, mem::zeroed, vec};
use indicatif::ProgressBar;
use na::{Dyn, SMatrix};
use nalgebra as na;
use serde::de::value;
use serde::{Serialize, Deserialize};
use serde_json::{json, Value};
use crate::datatypes::{State, Inputs};
use crate::{fc::*, graphical};
use crate::graphical::*;
use std::net::TcpStream;
use std::io::{Read, Write};


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
}

impl Sim {

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

    pub fn get_object(&mut self, id: usize) -> &mut Box<dyn Simulatable> {
        &mut self.objects[id]
    }

    pub fn export_current_state_to_json_str(&mut self) -> &str {
        todo!();
    }

    pub fn run_until(&mut self, end_time: f64) {
        self.end_time = end_time;
        let mut bar = ProgressBar::new((end_time / self.dt) as u64);
        let mut graphical_frame_counter: u64 = 0;
        while self.current_time < self.end_time {
            for i in 0..self.objects.len(){
                // Advance to next time step t_n+1, and iterate over each object to 
                let t = self.current_time;
                self.get_object(i).integrate_to_t(t);
                self.get_object(i).observe_position_and_rotation();
                // println!("{}", self.get_object(i).get_state());
            }


            
            if graphical_frame_counter >= self.frame_counter_update {
                self.datacom_update_packet();
                graphical_frame_counter = 0;
            }
            else {
                graphical_frame_counter += 1;
            }
            self.current_time = self.current_time + self.dt;
            self.steps += 1;
            bar.inc(1);
        }
    }

    pub fn add_object(&mut self, object: Box<dyn Simulatable>) {
        self.objects.push(object);
    }

    pub fn scene_initialization_to_datacom(&self) -> String {
        
        let mut json_str: String = "".to_string();
        
        for object in self.objects.iter() {
            json_str.push_str(object.json_object_initialization().to_string().as_str());
        }
        
        return json_str;
    }

    pub fn record_run(&self, folderpath: &str) {
        for object in (&self.objects).into_iter() {
            let filepath = "".to_owned() + folderpath+"/object_"+object.get_name()+".csv";
            object.export_data(&filepath);
        }
    }

    pub fn load_scenario(folderpath: &str) {
        todo!();
    }

    pub fn add_datacom_port(&mut self, new_port: String) {
        self.datacom_port = new_port;
    }

    pub fn initialize_datacom_json(&mut self) -> Value {
            let mut entities: Vec<serde_json::Value> = vec![];
            for i in 0..self.objects.len(){
                let temp = self.get_object(i).datacom_json_initialize();
                match &temp {
                    Some(Value) => entities.push(temp.expect("You shouldn't see this! ")),
                    _ => (),
                }
            }

            let datacom_packet = json!({
                "entities": entities
            });

            debug!("{}", datacom_packet.to_string());
            return datacom_packet;
    }

    pub fn datacom_update_packet(&mut self) -> Vec<Value> {
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
            debug!("{}", json!(commands).to_string());
            return commands;
    }

    pub fn datacom_send_packet(&self, addr: &str, data: &str) -> Result<(), Error> {

        let max_connection_attempts = 10;
        let mut num_attempts = 0;
        while max_connection_attempts > num_attempts {
            debug!("Connection attempt {}", num_attempts+1);
            let connection_result = TcpStream::connect(addr);


            match &connection_result {
                Ok(_) => {
                    debug!("Established connection. Transmitting data...");
                    let mut stream = connection_result.unwrap();
                    stream.write_all(&data.as_bytes()).unwrap();
                    debug!("Successfully transmitted packet");
                    return Ok(());
                },
                Err(_) => {
                    debug!("Connection attempt failed");
                    num_attempts+=1;
                }
            }
        }

        
        error!("Connection attempt timed out.");
        Err(Error)
    }

    pub fn datacom_start(&mut self, datacom_addr: &str) -> Result<(), Error> {
        // Set relevant data flags
        self.is_gui = true;

        // Generate initial data JSON and attempt to connect to DATACOM
        info!("Attempting to send initialization packet.");
        let initialization_packet = self.initialize_datacom_json();
        self.datacom_send_packet(datacom_addr, &initialization_packet.to_string())?;

        // Retrieve and send model files      
        

        Ok(())
    }

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
            is_gui: true,
            datacom_port: "".to_string(),
            frame_counter_update: 100,
        }
    }
    
}

pub trait Simulatable {
    fn get_name(&self) -> &str;
    fn integrate_to_t(&mut self, t: f64);
    fn observe_position_and_rotation(&mut self);
    fn get_position(&self) -> na::Point3<f64>;
    fn get_xdot(&self, x: &State, t: &f64) -> State;
    fn set_position(&mut self, new_position: na::Point3<f64>);
    fn get_rotation(&self) -> na::Point3<f64>;
    fn set_rotation(&mut self, new_rotation: na::Point3<f64>);
    fn get_state(&self) -> SMatrix<f64, 12, 1>;
    fn set_state(&mut self, new_x: State);
    fn get_component_forces(&mut self, t: f64) -> State;
    fn json_object_initialization(&self) -> sj::Value;
    fn status_to_json(&self) -> sj::Value;
    fn export_data(&self, filepath: &str);
    fn add_component(&mut self, new_component: Box<dyn ComponentPart>);
    // fn test();
    fn datacom_json_initialize(&self) -> Option<serde_json::Value>;
    fn datacom_json_command_step(&self) -> Option<Vec<Value>>;
    fn get_model_path(&self) -> &str;
    // fn add_flight_controller(&self, new_fc: Box<dyn FlightControl<U>>)
}

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
    obj_file: String,
}


impl<const U: usize> Vehicle<U> {
    pub fn new() -> Self {
        Vehicle{ name: "".to_string(), 
        id: 0, 
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
        data: DataLogger::<12, U>::new(),
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

        return self.fc.calculate_u(current_state)
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

    pub fn load_from_json(filepath: &str) -> Vehicle<U> {
        todo!();
    }

    pub fn load_from_json_string(json_unparsed: &str) -> Vehicle<U> {
        todo!();
    }

    pub fn load_from_json_parsed(json_parsed: sj::Value) -> Vehicle<U> {
        todo!();
    }

    pub fn add_flight_controller(&mut self, new_fc: Box<dyn FlightControl<U>>) {
        self.fc = new_fc;
    }

    pub fn datacom_position_command(&self) -> Value {
        let pos_cmd = json!({
            "targetEntityID": self.id,
            "commandType": "",
            "data": get_point_as_list(self.position),
        });
        return pos_cmd;
    }

    pub fn datacom_rotation_command(&self) -> Value {
        let rot_cmd = json!({
            "targetEntityID": self.id,
            "commandType": "",
            "data": get_point_as_list(self.rotation),
        });

        return rot_cmd
    }
}

impl<const U: usize> Simulatable for Vehicle<U> {

    fn get_name(&self) -> &str {
        &self.name
    }
    
    fn integrate_to_t(&mut self, t: f64) {
        let F_gravity = self.mass * 9.81 * z_acc_down(); 
        // debug!("Acceleration due to Gravity: {}", F_gravity);
        // let x_dot = self.get_xdot_A() + self.get_xdot_B(self.get_u(self.x));
        // debug!("U: {}", self.get_xdot_B(self.get_u(self.x)));
        // debug!("xdot: {}", x_dot);
        let dt = t - self.last_time;
        // self.x = self.x
        //  + dt*(
        //     x_dot
        //     + F_gravity/self.mass
        //  );
        self.x = self.integrator.integrate(
            |x, t| self.get_xdot(&x, &t), &self.x, &self.last_time, &t);
        self.last_time = t;
        self.data.record(t, self.get_state(), self.get_u(self.x))
    }

    fn get_xdot(&self, x: &State, t: &f64) -> State {
        let F_gravity = self.mass * 9.81 * z_acc_down(); 
        self.A*x
            + self.B*self.get_u(*x)
            + F_gravity/self.mass
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

        debug!("{}", json_str);

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

    fn export_data(&self, filepath: &str) {
        self.data.to_csv(filepath);
    }

    fn add_component(&mut self, new_component: Box<dyn ComponentPart>) {
        self.motors.push(new_component);
    }

    fn datacom_json_initialize(&self) -> Option<sj::Value> {
        let json_file = if self.is_graphical {
            Option::Some(json!({
                "Name": self.name,
                "id": self.id,
                "Position": [self.position[0], self.position[1], self.position[2]],
                "Rotation": [self.rotation[0], self.rotation[1], self.rotation[2]],
            }))
        }
        
        else {
            Option::None
        };

        // let json: Value = 
        return json_file;
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

    fn get_model_path(&self) -> &str {
        &self.obj_file
    }
}

impl<const U: usize> Default for Vehicle<U> {
    fn default() -> Self {
        Vehicle {
            name: "DEFAULT".to_string(),
            id: u64::MAX,
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
            data: DataLogger::<12, U>::new(),
            is_graphical: true,
            obj_file: "DEFAULT".to_string(),
        }
    }
}
pub enum PhysicsType {
    Static,
    StaticTrajectory,
    StateSpace,
}
// pub struct Rocket {
//     name: String,
//     mass: f64,
//     last_time: f64,
//     A: na::SMatrix<f64, 12, 12>,
//     x: State,
//     position: na::Point3<f64>,
//     rotation: na::Point3<f64>,
//     components: Vec<Box<dyn ComponentPart>>
// }

// impl Rocket {
//     pub fn new() -> Rocket {
//         Rocket{
//             name: "".to_string(),
//             mass: todo!(),
//             last_time: todo!(),
//             A: todo!(),
//             x: todo!(),
//             position: todo!(),
//             rotation: todo!(),
//             components: vec![]
//         }
//     }
// }

// impl Simulatable for Rocket {
//     fn get_name(&self) -> &str {
//         todo!()
//     }

//     fn integrate_to_t(&mut self, t: f64) {
//         let F_gravity = self.mass * 9.81 * z_down(); 
//         let x_dot = self.A*self.x;
//         // debug!("xdot: {}", x_dot);
//         let dt = t - self.last_time;
//         self.x = self.x
//          + x_dot*dt
//          + F_gravity/self.mass;
//         self.last_time = t;
//     }

//     fn observe_position_and_rotation(&mut self) {
//         self.position = na::Point3::new(self.x[0], self.x[1], self.x[2]);
//         self.rotation = na::Point3::new(self.x[6], self.x[7], self.x[8]);
//     }

//     fn get_xdot(&self, x: &State, t: &f64) -> State {
//         self.A*self.x
//     }

//     fn get_position(&self) -> na::Point3<f64> {
//         self.position
//     }

//     fn set_position(&mut self, new_position: na::Point3<f64>) {
//         self.position = new_position;
//     }

//     fn get_state(&self) -> SMatrix<f64, 12, 1> {
//         self.x
//     }

//     fn set_state(&mut self, new_x: State) {
//         self.x = new_x;
//     }

//     fn get_rotation(&self) -> na::Point3<f64> {
//         self.rotation
//     }

//     fn set_rotation(&mut self, new_rotation: na::Point3<f64>) {
//         self.rotation = new_rotation
//     }

//     fn get_component_forces(&mut self, t: f64) -> State {
//         let mut component_force_vec = na::SMatrix::<f64, 12, 1>::zeros();
//         for component in &mut self.components {
//             component.update_to_time(vec![], t);
//             component_force_vec = component_force_vec + component.get_force_on_parent();
//         };

//         return component_force_vec;
//     }

//     fn json_object_initialization(&self) -> sj::Value {
//         todo!()
//     }

//     fn status_to_json(&self) -> sj::Value {
//         // let json = json!(
//         //     "id": self.id
//         // )
//         todo!()
//     }

//     fn export_data(&self, filepath: &str) {
//         todo!();
//     }
// }


pub struct ElectricalMotor {
    J: f64,
    Kv: f64,
    Kt: f64,
    R: f64,
    L: f64,
    N: f64,
    omega: f64,
}

impl ElectricalMotor {
    pub fn new() -> ElectricalMotor {
        todo!();
    }

    pub fn new_from_json() -> ElectricalMotor {
        todo!();
    }

}

pub struct RocketMotor {
    ignited: IgnitionStatus,
    time_since_ignition: f64,
    burn_time_vec: Vec<f64>,
    thrust_curve: Vec<f64>
}

impl RocketMotor {

    pub fn new(burn_time_vec: Vec<f64>, thrust_curve: Vec<f64>) -> RocketMotor {

        RocketMotor {
            ignited: IgnitionStatus::Ready,
            time_since_ignition: 0.0,
            burn_time_vec: vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            thrust_curve: vec![1.0, 10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 2.5, 0.0]
        }
    }
}

// ###################

pub struct IdealThruster {
    time_const: f64,
    force: f64,
    set_force: f64,
    last_time: f64,
    position: na::SMatrix<f64, 3, 1>,
    orientation: na::UnitVector3<f64>,
    integrator: Integrator
}

impl IdealThruster {

    pub fn new() -> Self {
        IdealThruster {
            time_const: 0.0001,
            force: 0.0,
            set_force: 0.0,
            last_time: 0.0,
            position: na::SMatrix::zeros(),
            orientation: na::UnitVector3::new_unchecked(na::Vector3::new(0.0, 0.0, 1.0)),
            integrator: Integrator::RK4
        }
    }

    pub fn set_force(&mut self, new_force: f64) {
        self.set_force = new_force;
    }

    pub fn get_xdot(&self, y: &na::SMatrix<f64, 1, 1>, t: &f64) -> na::SMatrix<f64, 1, 1>{
        return (na::SMatrix::<f64, 1, 1>::new(self.force) - y)/self.time_const;
    }
}

impl ComponentPart for IdealThruster {
    fn get_force_on_parent(&self) -> State {
        let force_vec = self.force * *self.orientation;
        let torque_vec = force_vec.cross(&self.position);
        let force_and_torque = State::from_row_slice(&[
            0.0, 
            0.0,
            0.0,
            force_vec[0],
            force_vec[1],
            force_vec[2],
            0.0,
            0.0,
            0.0,
            torque_vec[0],
            torque_vec[1],
            torque_vec[2]
        ]);
        return force_and_torque;
    }

    fn update_to_time(&mut self, y: Vec<f64>, t: f64) {
        self.force = self.set_force;
    }

    fn set_force(&mut self, new_force: f64) {
        self.set_force = new_force;
    }
}

enum IgnitionStatus {
    Ready,
    Active,
    Spent
}
pub enum Status {
    Active,
    Inactive
}

impl Status {
    pub fn activate() {
        todo!()
    }
}

pub enum Integrator {
    ForwardEuler,
    RK4,
}

impl Integrator {
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

pub trait ComponentPart {
    fn update_to_time(&mut self, y: Vec<f64>, t: f64);
    fn get_force_on_parent(&self) -> SMatrix<f64, 12, 1>;
    fn set_force(&mut self, new_force:f64);
}

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

fn forward_matrix(dt: f64) -> SMatrix<f64, 12, 12> {
    na::SMatrix::<f64, 12, 12>::from_row_slice(&[
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 

        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 

        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 

        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
    ])
}


// #####################


pub struct DataLogger<const S: usize, const U: usize> {
    time_vector: Vec<f64>,
    state_vector: Vec<na::SMatrix<f64, S, 1>>,
    input_vector: Vec<Inputs<U>>
}

impl<const S: usize, const U:usize> DataLogger<S, U> {

    pub fn new() -> Self {
        DataLogger{
            time_vector: vec![],
            state_vector: vec![],
            input_vector: vec![]
        }
    }

    pub fn record(&mut self, time: f64, state: na::SMatrix<f64, S, 1>, input: Inputs<U>) {
        self.time_vector.push(time);
        self.state_vector.push(state);
        self.input_vector.push(input);

    }

    pub fn to_csv(&self, filepath: &str) {
        // Setup
        let mut csv: String = "Time,X Position,Y Position,Z Position,x_vel,y_vel,z_vel,Pitch,Roll,Yaw,pitch_vel,roll_vel,yaw_vel".to_string();
        for i in (&self.input_vector[0]) {
            csv.push_str(&format!(",U_{}", i));
        }
        csv.push_str("\n");

        // Iterate over all datapoints
        for (i, _) in self.time_vector.iter().enumerate() {
            let mut local_str = "".to_string();
            local_str = local_str + &format!("{:?}", self.time_vector[i]); // Time
            for state_el in (&self.state_vector[i]).into_iter() {
                local_str.push_str(&format!(",{:?}", state_el));
            }

            for input in (&self.input_vector[i]).into_iter() {
                local_str.push_str(&format!(",{:?}", input));
            }
            local_str.push_str("\n");
            csv.push_str(&local_str);
        }

        // Write to file
        debug!("{}", filepath);
        std::fs::write(filepath, csv).unwrap();
 
    }

    pub fn how_many_steps(&self) -> u64 {
        self.time_vector.len() as u64
    }
} 

fn get_point_as_list(point: na::Point3<f64>) -> [f64; 3] {
    return [point[0], point [1], point[2]];
}