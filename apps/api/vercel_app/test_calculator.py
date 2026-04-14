"""
Unit tests for the Calculator application
Tests all calculator operations to ensure they work correctly
"""

import unittest
from calculator import Calculator


class TestCalculator(unittest.TestCase):
    """Test cases for the Calculator class"""

    def setUp(self):
        """Set up a fresh calculator instance before each test"""
        self.calc = Calculator()

    def test_add(self):
        """Test addition operation"""
        self.assertEqual(self.calc.add(5, 3), 8)
        self.assertEqual(self.calc.add(-5, 3), -2)
        self.assertEqual(self.calc.add(0, 0), 0)
        self.assertEqual(self.calc.add(1.5, 2.5), 4.0)

    def test_subtract(self):
        """Test subtraction operation"""
        self.assertEqual(self.calc.subtract(10, 3), 7)
        self.assertEqual(self.calc.subtract(-5, 3), -8)
        self.assertEqual(self.calc.subtract(5, 5), 0)
        self.assertEqual(self.calc.subtract(5.5, 2.5), 3.0)

    def test_multiply(self):
        """Test multiplication operation"""
        self.assertEqual(self.calc.multiply(5, 3), 15)
        self.assertEqual(self.calc.multiply(-5, 3), -15)
        self.assertEqual(self.calc.multiply(0, 100), 0)
        self.assertEqual(self.calc.multiply(2.5, 4), 10.0)

    def test_divide(self):
        """Test division operation"""
        self.assertEqual(self.calc.divide(10, 2), 5)
        self.assertEqual(self.calc.divide(9, 3), 3)
        self.assertEqual(self.calc.divide(7, 2), 3.5)
        self.assertEqual(self.calc.divide(-10, 2), -5)

    def test_divide_by_zero(self):
        """Test division by zero raises error"""
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)

    def test_power(self):
        """Test power operation"""
        self.assertEqual(self.calc.power(2, 3), 8)
        self.assertEqual(self.calc.power(5, 0), 1)
        self.assertEqual(self.calc.power(10, 2), 100)
        self.assertEqual(self.calc.power(2, -1), 0.5)

    def test_square_root(self):
        """Test square root operation"""
        self.assertEqual(self.calc.square_root(9), 3)
        self.assertEqual(self.calc.square_root(16), 4)
        self.assertEqual(self.calc.square_root(0), 0)
        self.assertAlmostEqual(self.calc.square_root(2), 1.414213562, places=5)

    def test_square_root_negative(self):
        """Test square root of negative number raises error"""
        with self.assertRaises(ValueError):
            self.calc.square_root(-5)

    def test_history(self):
        """Test calculation history"""
        self.calc.add(5, 3)
        self.calc.subtract(10, 2)
        history = self.calc.get_history()
        self.assertEqual(len(history), 2)
        self.assertIn("5 + 3 = 8", history[0])
        self.assertIn("10 - 2 = 8", history[1])

    def test_clear_history(self):
        """Test clearing history"""
        self.calc.add(5, 3)
        self.calc.clear_history()
        self.assertEqual(len(self.calc.get_history()), 0)

    def test_get_result(self):
        """Test getting the last result"""
        self.calc.add(5, 3)
        self.assertEqual(self.calc.get_result(), 8)
        self.calc.multiply(2, 3)
        self.assertEqual(self.calc.get_result(), 6)


if __name__ == "__main__":
    unittest.main()
