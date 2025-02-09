from create_scenario import *
import torch 
import torch.nn as nn
from tqdm import tqdm

class RLController(nn.Module):

    def __init__(self, input_dim=12, hidden_layer=144, output_dim=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)

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
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}/{num_epochs}: Average Loss = {avg_loss:,.4f}")

        

if __name__ == "__main__":
    primer_job = basic_job()
    primer_job.export_job()
    # sim_data_vector = call_sim()
    sim_data_vector = load_all_simulation_runs("data/todo/default_name/output")
    model = RLController(12, 500, 8)
    fc = FlightComputer(0.001, [], model)
    job = nn_att_2(fc)
    job.export_job()
    params = job.objects[0].fc.K.network.parameters()
    optimizer = torch.optim.Adam(params, lr=1E-3)
    # optimizer = torch.optim.SGD(params, lr=1E-5)
    train_from_example_K(job, model, optimizer, sim_data_vector, num_epochs=50, batch_size=4096)