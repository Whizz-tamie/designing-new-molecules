import unittest
import torch
from src.models.pgfs.models.knn import KNN

class TestKNN(unittest.TestCase):
    
    def setUp(self):
        self.knn = KNN(k=2)
        self.action = torch.tensor([1.0, 2.0, 3.0])
        self.valid_reactants_vectors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0]),
            torch.tensor([7.0, 8.0, 9.0])
        ]
    
    def test_find_neighbors_valid_input(self):
        neighbors = self.knn.find_neighbors(self.action, self.valid_reactants_vectors)
        self.assertEqual(len(neighbors), 2)
        self.assertTrue(all(isinstance(n, torch.Tensor) for n in neighbors))
    
    def test_find_neighbors_invalid_action_type(self):
        with self.assertRaises(ValueError):
            self.knn.find_neighbors([1.0, 2.0, 3.0], self.valid_reactants_vectors)
    
    def test_find_neighbors_invalid_reactant_vectors_type(self):
        with self.assertRaises(ValueError):
            self.knn.find_neighbors(self.action, [[1.0, 2.0, 3.0]])
    
    def test_find_neighbors_empty_reactants(self):
        with self.assertRaises(ValueError):
            self.knn.find_neighbors(self.action, [])
    
    def test_find_neighbors_single_reactant(self):
        self.knn.k = 1
        neighbors = self.knn.find_neighbors(self.action, [torch.tensor([1.0, 2.0, 3.0])])
        self.assertEqual(len(neighbors), 1)
        self.assertTrue(torch.equal(neighbors[0], torch.tensor([1.0, 2.0, 3.0])))

    def test_find_neighbors_k_greater_than_reactants(self):
        self.knn.k = 5
        neighbors = self.knn.find_neighbors(self.action, self.valid_reactants_vectors)
        self.assertEqual(len(neighbors), 3)  # Should return all available reactants since k > len(reactants)

    def test_find_neighbors_action_on_gpu(self):
        if torch.cuda.is_available():
            action_gpu = self.action.cuda()
            neighbors = self.knn.find_neighbors(action_gpu, self.valid_reactants_vectors)
            self.assertEqual(len(neighbors), 2)
            self.assertTrue(all(isinstance(n, torch.Tensor) for n in neighbors))

if __name__ == '__main__':
    unittest.main()
