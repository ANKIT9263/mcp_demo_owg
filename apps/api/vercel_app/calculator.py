"""
Basic Calculator Application
A simple calculator that performs basic arithmetic operations
"""


class Calculator:
    """A simple calculator class for basic arithmetic operations"""

    def __init__(self):
        """Initialize the calculator"""
        self.result = 0
        self.history = []

    def add(self, a: float, b: float) -> float:
        """
        Add two numbers
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            The sum of a and b
        """
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        self.result = result
        return result

    def subtract(self, a: float, b: float) -> float:
        """
        Subtract two numbers
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            The difference of a and b
        """
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        self.result = result
        return result

    def multiply(self, a: float, b: float) -> float:
        """
        Multiply two numbers
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            The product of a and b
        """
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        self.result = result
        return result

    def divide(self, a: float, b: float) -> float:
        """
        Divide two numbers
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            The quotient of a and b
            
        Raises:
            ValueError: If attempting to divide by zero
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        self.result = result
        return result

    def power(self, a: float, b: float) -> float:
        """
        Raise a number to a power
        
        Args:
            a: Base number
            b: Exponent
            
        Returns:
            a raised to the power of b
        """
        result = a ** b
        self.history.append(f"{a} ** {b} = {result}")
        self.result = result
        return result

    def square_root(self, a: float) -> float:
        """
        Calculate the square root of a number
        
        Args:
            a: The number to get the square root of
            
        Returns:
            The square root of a
            
        Raises:
            ValueError: If attempting to get square root of negative number
        """
        if a < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = a ** 0.5
        self.history.append(f"sqrt({a}) = {result}")
        self.result = result
        return result

    def get_history(self) -> list:
        """
        Get the calculation history
        
        Returns:
            List of all calculations performed
        """
        return self.history

    def clear_history(self) -> None:
        """Clear the calculation history"""
        self.history = []

    def get_result(self) -> float:
        """
        Get the last result
        
        Returns:
            The last calculated result
        """
        return self.result
