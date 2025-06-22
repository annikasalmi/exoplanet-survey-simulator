"""
Basic tests to verify the testing setup works correctly.
"""

import unittest
import numpy as np


class TestBasicSetup(unittest.TestCase):
    """Basic tests to verify the testing environment."""
    
    def test_numpy_import(self):
        """Test that numpy is working."""
        arr = np.array([1, 2, 3, 4, 5])
        self.assertEqual(len(arr), 5)
        self.assertEqual(arr.sum(), 15)
        
    def test_basic_math(self):
        """Test basic mathematical operations."""
        self.assertEqual(2 + 2, 4)
        self.assertAlmostEqual(np.pi, 3.14159, places=4)
        
    def test_physics_constants_import(self):
        """Test that physics constants can be imported."""
        try:
            from tools.physics_constants import h, c, k, sigma
            self.assertGreater(h, 0)
            self.assertGreater(c, 0)
            self.assertGreater(k, 0)
            self.assertGreater(sigma, 0)
        except ImportError as e:
            self.fail(f"Failed to import physics constants: {e}")


if __name__ == '__main__':
    unittest.main() 