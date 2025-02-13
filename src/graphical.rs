use sj::{json, Value};

/// Struct for containing graphical data
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
    /// Create new graphical struct
    pub fn new(name: &str, filepath: &str, color: [f64; 4]) -> Self {
        GraphicalData {
            name: name.to_string(),
            filepath: filepath.to_string(),
            color: color,
            ..Default::default()
        }
    }

    /// Load a graphical struct from JSON
    pub fn from_json(json_file: &Value) -> Self {
        let name = json_file["Name"].as_str().unwrap();
        let object_file_path = json_file["ObjectFilePath"].as_str().unwrap();
        // let scale: [f64; 3]=json_file["Scale"].as_array().unwrap().into_iter().map(|x| x.as_f64().unwrap()).collect();

        let pos_temp: Vec<_> = json_file["Position"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let mut pos_vec = [pos_temp[0], pos_temp[1], pos_temp[2]];

        let or_temp: Vec<_> = json_file["Orientation"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let mut or_vec = [or_temp[0], or_temp[1], or_temp[2]];

        let rot_temp: Vec<_> = json_file["Rotation"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let mut rot_vec = [rot_temp[0], rot_temp[1], rot_temp[2]];

        let scale_temp: Vec<_> = json_file["Scale"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let mut scale_vec = [scale_temp[0], scale_temp[1], scale_temp[2]];

        let color_temp: Vec<_> = json_file["Color"]
            .as_array()
            .unwrap()
            .into_iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let mut color_vec = [color_temp[0], color_temp[1], color_temp[2], color_temp[3]];

        GraphicalData { name: name.to_string(), filepath: object_file_path.to_string(), relative_position: pos_vec, orientation: or_vec, rotation: rot_vec, color: color_vec, scale: scale_vec }

    }

    /// Return model as a JSON Value struct
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

    /// Get filepath of the model
    pub fn get_model_path(&self) -> &str {
        &self.filepath
    }

    /// Set model scale
    pub fn set_scale(&mut self, new_scale: [f64; 3]) {
        self.scale=new_scale;
        debug!("New Scale: {}", self.scale[0]);
    }
}