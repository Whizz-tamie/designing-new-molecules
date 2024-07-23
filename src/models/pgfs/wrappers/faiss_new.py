import logging
import random

import faiss
import faiss.contrib.torch_utils
import gymnasium as gym
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED

# Configure the logger
logger = logging.getLogger(__name__)


class KNNWrapper(gym.ActionWrapper):
    def __init__(self, env, enabled=True):
        super(KNNWrapper, self).__init__(env)
        logger.info("New KNNWrapper initiated...")

        self.enabled = enabled
        self.reactants = (
            self.env.unwrapped.reactants  # type: ignore
        )  # Access reactants from environment
        self.knn_indices = {}  # Dictionary to store FAISS indices
        self._initialize_resources()

    def _initialize_resources(self):
        try:
            self.res = faiss.StandardGpuResources()  # Use standard GPU resources
            self.use_gpu = True
            logger.info("Using GPU resources for FAISS.")
        except Exception as e:
            self.res = None  # Ensure resource is None if initialization fails
            self.use_gpu = False
            logger.warning(
                "GPU resources not available, falling back to CPU. Error: " + str(e)
            )

    def _initialize_index_for_template(self, template_index):
        """Lazy initialization of FAISS index for a given template with logging of data added."""
        if template_index not in self.knn_indices:
            if self.env.unwrapped.templates[template_index]["type"] == "bimolecular":
                logger.debug(
                    "Template %s is bimolecular. If second reactants avaliable, initialise FAISS index.",
                    template_index,
                )
                valid_reactants = (
                    self.env.unwrapped.reaction_manager.get_valid_reactants(
                        template_index
                    )
                )

                if valid_reactants:
                    # Create tensor of fingerprints
                    reactant_fingerprints = torch.stack(
                        [
                            torch.tensor(self.reactants[reactant], dtype=torch.float32)
                            for reactant in valid_reactants
                        ]
                    )

                    # Move to GPU if enabled
                    if self.use_gpu:
                        reactant_fingerprints = reactant_fingerprints.cuda()

                    # Create the FAISS index
                    index = (
                        faiss.GpuIndexFlatL2(self.res, reactant_fingerprints.shape[1])
                        if self.use_gpu
                        else faiss.IndexFlatL2(reactant_fingerprints.shape[1])
                    )

                    # Convert tensor to numpy array and add to index
                    reactant_fingerprints_np = (
                        reactant_fingerprints.cpu().numpy()
                        if self.use_gpu
                        else reactant_fingerprints.numpy()
                    )
                    index.add(reactant_fingerprints_np)  # Add data to the index

                    # Store the index
                    self.knn_indices[template_index] = index
                    logger.info(
                        "FAISS index for template %s is initialised on %s with %s reactants.",
                        template_index,
                        "GPU" if self.use_gpu else "CPU",
                        index.ntotal,
                    )
                else:
                    logger.warning(
                        "No valid reactants found for template %s; index not created.",
                        template_index,
                    )
            else:
                logger.info(
                    "Template %s is unimolecular; no index needed.", template_index
                )

    def _ensure_index_initialized(self, template_index):
        if template_index not in self.knn_indices:
            logger.debug(
                f"Template {template_index} is not in knn_indices. Ensuring it is created..."
            )
            self._initialize_index_for_template(template_index)

    def action(self, action):
        if self.enabled:
            template_one_hot, reactant_vector = action
            logger.debug("r2_tanh: %s", torch.unique(reactant_vector))

            # Ensure the output is binary
            reactant_vector = (reactant_vector >= 0).float()
            logger.debug(
                "r2_vector converted to binary: %s, shape: %s",
                torch.unique(reactant_vector),
                reactant_vector.shape,
            )

            template_index = torch.argmax(template_one_hot).item()
            self._ensure_index_initialized(template_index)
            action = self._process_knn_search(
                self.knn_indices.get(template_index), reactant_vector, template_index
            )
            logger.info(
                "KNN Wrapper processed action for template %s...", template_index
            )
        else:
            template_one_hot, r2 = action
            template_index = torch.argmax(template_one_hot).item()
            action = (
                template_index,
                (r2 if isinstance(r2, str) else None),
            )
            logger.info(
                "KNN Wrapper disabled. Passing action through - Template index: %s, R2: %s",
                template_index,
                r2,
            )
        return action

    def _process_knn_search(
        self, knn_index, reactant_vector, template_index, k=5, epsilon=0.1
    ):
        logger.debug("Initiating KNN search for template %s", template_index)

        if knn_index is None or torch.all(reactant_vector.eq(0)):
            logger.info(
                "No valid KNN index or second reactant vector for template %s. Default action applied.",
                template_index,
            )
            return (template_index, None)

        try:
            query_vector = (
                reactant_vector.cuda()
                if self.use_gpu
                else reactant_vector.cpu().numpy()
            )
            distances, indices = knn_index.search(query_vector, k)
            logger.debug("indices: %s", indices.size)

            if indices.size == 0:
                logger.warning(
                    "No neighbors found for template %s. Check if the FAISS index is populated.",
                    template_index,
                )
                return (template_index, None)

            valid_reactants = self.env.unwrapped.reaction_manager.get_valid_reactants(
                template_index
            )
            top_reactants = [valid_reactants[idx] for idx in indices[0]]

            # Epsilon-greedy selection
            if random.random() < epsilon:
                # Exploration: Select a random reactant from the top K neighbors
                selected_reactant = random.choice(top_reactants)
                logger.info(
                    "Exploration: Randomly selected reactant for template %s: %s",
                    template_index,
                    selected_reactant,
                )
            else:
                # Exploitation: Select the best reactant based on QED or other criteria
                best_reactant = None
                best_score = -np.inf
                for reactant in top_reactants:
                    mol = Chem.MolFromSmiles(reactant)
                    qed = QED.qed(mol) if mol else 0
                    # Combine QED with distance-based score (or any other criteria)
                    score = qed - distances[0][top_reactants.index(reactant)]
                    if score > best_score:
                        best_score = score
                        best_reactant = reactant

                selected_reactant = best_reactant
                logger.info(
                    "Exploitation: Best reactant selected for template %s: %s",
                    template_index,
                    selected_reactant,
                )

            return (template_index, selected_reactant)

        except Exception as e:
            logger.error(
                "Unexpected error during KNN search for template %s: %s",
                template_index,
                str(e),
            )
            return (template_index, None)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
