from create_scenario import *
import torch 
import torch.nn as nn
from tqdm import tqdm

class RLController(nn.Module):

    def __init__(self, input_dim=12, hidden_layer=144, output_dim=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_layer),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, hidden_layer, bias=False),
            # nn.LeakyReLU(),
            nn.Linear(hidden_layer, output_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.network(x)
    
def generate_data(num_points: int):
    K = np.loadtxt("data/todo/default_name/objects/blizzard/Klqr.csv", dtype=float, delimiter=',')
    mixer = np.array([[0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                [0, 0, 0, 0, 0, 0, 0, 0],  # X Forces
                [0, 0, 0, 0, 0, 0, 0, 0],  # Y Forces
                [1, 1, 1, 1, 1, 1, 1, 1],  # Z Forces
                [0, 0, 0, 0, 0, 0, 0, 0],  # Empty
                [0, 0, 0, 0, 0, 0, 0, 0],  # Empty 
                [0, 0, 0, 0, 0, 0, 0, 0],  # Empty 
                [3, -3, 3, -3, 3, -3, 3, -3],  # X Moments (Roll)
                [2.5, 2.5, -2.5, -2.5, 2.5, 2.5, -2.5, -2.5],  # Y Moments (Pitch)
                [-1, 1, 1, -1, 1, -1, -1, 1]], dtype=float)  # Z Moments (Yaw)
    
    I = np.array([[600, 0, 0],
              [0, 800, 0],
              [0, 0, 800]])
    mass = 2200
    
    B = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # x'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # y'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # z'
                           [0, 0, 0, 1/mass, 0, 0, 0, 0, 0, 0, 0, 0],       # x''
                           [0, 0, 0, 0, 1/mass, 0, 0, 0, 0, 0, 0, 0],       # y''
                           [0, 0, 0, 0, 0, 1/mass, 0, 0, 0, 0, 0, 0],       # z''
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # phi'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # theta'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # psi'
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # phi''
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],            # theta''
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],           # psi''
                          dtype=float)
    
    B[9:12, 9:12] = np.linalg.inv(I)
    B = B @ mixer
    # print(B)

    K = np.loadtxt("data/todo/default_name/objects/blizzard/Klqr.csv", dtype=float, delimiter=',')

    K[9,1] = -0.1*K[9,6]
    K[9,4] = -0.3*K[9,9]

    K[10,0] = 0.1*K[10,7]
    K[10,3] = 0.3*K[10,10]

    inputs = np.zeros((12,1))
    outputs = np.zeros((8,1))

    K = mixer.transpose()@K
    for i in tqdm(range(num_points), desc="Point Generation"):
        x = np.random.rand(12, 1)
        u = - K @ x
        # print(u.shape)
        inputs = np.concatenate([inputs, x], axis = 1)
        outputs = np.concatenate([outputs, u], axis=1)

    # print(inputs.transpose().shape, outputs.transpose().shape)
    return inputs.transpose(), outputs.transpose()

def train_from_model(optimizer, model, num_points=1000, num_epochs=50, batch_size=32):
    states, actions = generate_data(num_points=num_points)
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    model.train(True)
    n_samples = len(states)
    print(f"Dataset size: {n_samples:,}")
    n_batches = n_samples // batch_size
    

    for epoch in range(num_epochs):
        total_loss = 0.0        
        indices = torch.randperm(n_samples)
        
        for batch in tqdm(range(n_batches)):
            optimizer.zero_grad()
            
            # Get batch indices
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]

            model_estimates = model(batch_states)
            # Compute loss - ensure dimensions match
            # print(model_estimates)
            loss = nn.MSELoss()(model_estimates, batch_actions)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # Print epoch statistics
            # print(f"Batch {batch}/{n_batches} loss: {loss.item()}")
             
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}/{num_epochs}: Average Loss = {avg_loss:,.4f}")
    

def train_from_example_K(job, model, optimizer, training_data: [SimObjectOutput], *, num_epochs=100, batch_size=256):
    print("Training start!")
    states, next_states, actions = get_data_from_sim_run_list(sim_data_vector)
    states = torch.tensor(states, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    model.train(True)
    n_samples = len(states)
    print(f"Dataset size: {n_samples:,}")
    n_batches = n_samples // batch_size
    

    for epoch in range(num_epochs):
        total_loss = 0.0        
        indices = torch.randperm(n_samples)
        
        for batch in tqdm(range(n_batches)):
            optimizer.zero_grad()
            
            # Get batch indices
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_states = states[batch_indices]
            batch_next_states = next_states[batch_indices]
            batch_actions = actions[batch_indices]

            model_estimates = model(batch_states)
            # Compute loss - ensure dimensions match
            # print(model_estimates)
            loss = nn.MSELoss()(model_estimates, batch_actions)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # Print epoch statistics
            # print(f"Batch {batch}/{n_batches} loss: {loss.item()}")
             
        avg_loss = total_loss / n_samples
        print(f"Epoch {epoch + 1}/{num_epochs}: Average Loss = {avg_loss:,.4f}")

def train_PPO_strategy():
    pass

if __name__ == "__main__":
    primer_job = step_response_timing_test()
    primer_job.export_job()
    sim_data_vector = call_sim()

    plot_run(read_from_csv("data/todo/default_name/output/blizzard_hover_test/object_0_blizzard.csv"))
    # sim_data_vector = load_all_simulation_runs("data/todo/default_name/output")
    model = RLController(12, 500, 8)
    fc = FlightComputer(0.001, [], model)
    job = step_response_timing_test(fc=fc)
    job.objects[0].fc.set_NN_filepath("data/todo/default_name/objects/blizzard/blizzard.onnx")
    job.export_job()
    params = job.objects[0].fc.K.network.parameters()
    optimizer = torch.optim.Adam(params, lr=1E-3)
    # # optimizer = torch.optim.SGD(params, lr=1E-5)
    train_from_example_K(primer_job, model, optimizer, sim_data_vector, num_epochs=1000, batch_size=4096)