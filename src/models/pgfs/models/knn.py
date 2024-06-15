import torch

class KNN:
    def __init__(self, k=1):
        self.k = k

    def find_neighbors(self, action, valid_reactants):
        valid_reactants_vectors = list(dict(valid_reactants).values())
        valid_reactants_uuid = list(dict(valid_reactants).keys())

        if not isinstance(action, torch.Tensor):
            raise ValueError("Action must be a torch.Tensor")
        
        if not all(isinstance(vec, torch.Tensor) for vec in valid_reactants_vectors):
            raise ValueError("All reactant vectors must be torch.Tensor")
        
        if len(valid_reactants_vectors) == 0:
            raise ValueError("The list of valid reactants vectors is empty")

        # Ensure action is on the CPU
        action_vector = action.detach().cpu()

        # Stack all reactant vectors into a single tensor
        reactant_vectors = torch.stack(valid_reactants_vectors)

        # Compute distances between action and all reactant vectors
        distances = torch.cdist(action_vector.unsqueeze(0), reactant_vectors, p=2).squeeze(0)

        # Ensure k does not exceed the number of reactants
        k = min(self.k, len(valid_reactants_vectors))

        # Get the indices of the k smallest distances
        _, indices = torch.topk(distances, k, largest=False)

        # Convert indices to a flat list of integers
        indices = indices.flatten().tolist()

        # Debug print statement
        print(f"indices after conversion: {indices}")

        # Get the k-nearest neighbors uuid
        nearest_neighbors = [valid_reactants_uuid[i] for i in indices]

        return nearest_neighbors
