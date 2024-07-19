# knn.py

import gymnasium as gym
import numpy as np
from sklearn.neighbors import NearestNeighbors


class KNNWrapper(gym.ActionWrapper):
    def __init__(self, env, reactants):
        super(KNNWrapper, self).__init__(env)
        self.reactants = (
            reactants  # Dictionary of reactants with precomputed fingerprints
        )
        # Cache for storing KNN models for each template index
        self.knn_models = {}

    def action(self, action):
        template_one_hot, reactant_vector = action
        template_index = np.argmax(template_one_hot)

        # Fetch valid reactant vectors for the selected template index
        if template_index not in self.knn_models:
            valid_reactants = self.env.reaction_manager.get_valid_reactants(
                template_index
            )
            if valid_reactants:
                valid_reactant_vectors = [
                    self.reactants[idx] for idx in valid_reactants
                ]
                self.knn_models[template_index] = NearestNeighbors(n_neighbors=1).fit(
                    valid_reactant_vectors
                )
            else:
                self.knn_models[template_index] = None

        knn = self.knn_models[template_index]

        # If a second reactant vector is provided and valid reactants exist, perform KNN search
        if reactant_vector is not None and knn:
            reactant_vector = np.array(reactant_vector).reshape(1, -1)
            nearest_index = knn.kneighbors(reactant_vector, return_distance=False)[0][0]
            selected_reactant_index = valid_reactants[nearest_index]
            modified_action = (template_index, selected_reactant_index)
        else:
            # No valid second reactant needed or possible
            modified_action = (template_index, None)

        return modified_action
