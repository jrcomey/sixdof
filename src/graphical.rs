use tobj;
use serde::{Serialize};

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f64; 3],
}
impl Vertex {
    pub fn newtwo(x: f64, y: f64) -> Vertex {
        Vertex { position: [x, y, 0.0] }
    }

    pub fn new(x: f64, y: f64, z: f64) -> Vertex {
        Vertex { position: [x, y, z] }
    }
}


#[derive(Copy, Clone)]
pub struct Normal {
    normal: [f64; 3],
}

impl Normal {
    pub fn new(x: f64, y: f64, z: f64) -> Normal {
        Normal { normal: [x, y, z] }
    }
}
pub struct WireframeObject {
    // Define the structure for your wireframe representation
    // This could include vertex data, indices, color, etc.
    // ...

    positions:  Vec<Vertex>,
    normals:    Vec<Normal>,
    indices:    Vec<u32>,
    color:      na::base::Vector4<f32>
}

impl WireframeObject {
    pub fn load_wireframe_from_obj(filepath: &str, colorvec: na::base::Vector4<f32>) -> WireframeObject {
        let file = tobj::load_obj(filepath, &tobj::GPU_LOAD_OPTIONS);
        assert!(file.is_ok());
        
        let (models, _) = file.unwrap();
        let mesh = &models[0].mesh;
    
        WireframeObject{
            positions: WireframeObject::convert_to_vertex_struct(&mesh.positions),
            normals: WireframeObject::convert_to_normal_struct(&mesh.normals),
            indices: mesh.indices.clone(),
            color: colorvec
        }
    }

    pub fn convert_to_vertex_struct (target: &Vec<f32>) -> Vec<Vertex>{
        // Have to pass a reference (&) to the target. 
        // Datatype does not support copying, and Rust creates a copy of the orignal function arguments. 
        // Original copy not modified, so fine
        let mut vertex_array: Vec<Vertex> = std::vec::Vec::new();
        for i in 0..target.len()/3 {
            vertex_array.push(Vertex::new(target[3*i+0] as f64, target[3*i+1] as f64, target[3*i+2] as f64));
        };
        return vertex_array;
    }
    
    pub fn convert_to_normal_struct (target: &Vec<f32>) -> Vec<Normal>{
        // Have to pass a reference (&) to the target. 
        // Datatype does not support copying, and Rust creates a copy of the orignal function arguments. 
        // Original copy not modified, so fine
        let mut normal_array: Vec<Normal> = std::vec::Vec::new();
        for i in 0..target.len()/3 {
            normal_array.push(Normal::new(target[3*i+0] as f64, target[3*i+1] as f64, target[3*i+2] as f64));
        }
        return normal_array;
    }

    pub fn new(positions: Vec<Vertex>, normals: Vec<Normal>, indices: Vec<u32>, color: na::base::Vector4<f32>) -> WireframeObject {
        WireframeObject { positions: positions, normals: normals, indices: indices, color: color }
    }

    pub fn change_color(&mut self, new_color: na::Vector4<f32>) {
        self.color = new_color;
    }

    pub fn get_color(&mut self) -> na::Vector4<f32> {
        self.color
    }

}