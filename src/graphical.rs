use sj::{json, Value};

pub struct GraphicalData {
    pub name: String,
    pub filepath: String,
    pub relative_position: [f64; 3],
    pub orientation: [f64; 3],
    pub rotation: [f64; 3],
    pub color: [f64; 4],
    pub scale: [f64; 3],
}

impl Default for GraphicalData {
    fn default() -> Self {
        GraphicalData {
            name: "DEFAULT_MODEL_NAME".to_string(),
            filepath: "data/test_object/test.obj".to_string(),
            relative_position: [0.0, 0.0, 0.0],
            orientation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            color: [1.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0]
        }
    }
}

impl GraphicalData {

    pub fn new(name: &str, filepath: &str, color: [f64; 4]) -> Self {
        GraphicalData {
            name: name.to_string(),
            filepath: filepath.to_string(),
            color: color,
            ..Default::default()
        }
    }

    pub fn get_model_value(&self) -> Value {
        json!({
            "Name": self.name,
            "ObjectFilePath": self.filepath,
            "Position": self.relative_position,
            "Orientation": self.orientation,
            "Rotation": self.rotation,
            "Color": self.color,
            "Scale": self.scale
        })
    }

    pub fn get_model_path(&self) -> &str {
        &self.filepath
    }

    pub fn set_scale(&mut self, new_scale: [f64; 3]) {
        self.scale=new_scale;
        debug!("New Scale: {}", self.scale[0]);
    }
}