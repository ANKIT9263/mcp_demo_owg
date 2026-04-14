# Basic Calculator Application

A simple, well-structured calculator application written in Python. This application provides both a command-line interface and a reusable Calculator class for performing basic arithmetic operations.

## Features

- **Addition**: Add two numbers
- **Subtraction**: Subtract two numbers
- **Multiplication**: Multiply two numbers
- **Division**: Divide two numbers (with zero-check)
- **Power**: Raise a number to a power
- **Square Root**: Calculate the square root of a number
- **Calculation History**: View and clear calculation history
- **Interactive CLI**: User-friendly command-line interface

## Project Structure

```
apps/api/vercel_app/
├── calculator.py         # Core Calculator class with all operations
├── main.py              # CLI application entry point
├── test_calculator.py   # Unit tests for the calculator
└── README.md            # This file
```

## Files

### `calculator.py`
Contains the main `Calculator` class with the following methods:
- `add(a, b)` - Addition
- `subtract(a, b)` - Subtraction
- `multiply(a, b)` - Multiplication
- `divide(a, b)` - Division with error handling
- `power(a, b)` - Exponentiation
- `square_root(a)` - Square root with error handling
- `get_history()` - Retrieve calculation history
- `clear_history()` - Clear the history
- `get_result()` - Get the last result

### `main.py`
Interactive command-line application that provides a menu-driven interface for using the calculator. Users can:
1. Perform calculations
2. View calculation history
3. Clear history
4. Exit the application

### `test_calculator.py`
Comprehensive unit tests covering:
- All arithmetic operations
- Error handling (division by zero, square root of negative)
- History functionality
- Result tracking

## Usage

### Running the CLI Application

```bash
python main.py
```

This will start an interactive menu where you can select operations and enter numbers.

### Running Tests

```bash
python -m unittest test_calculator.py
```

Or to run with verbose output:

```bash
python -m unittest test_calculator.py -v
```

### Using the Calculator Class in Your Code

```python
from calculator import Calculator

calc = Calculator()

# Add two numbers
result = calc.add(5, 3)  # Returns 8

# View history
print(calc.get_history())  # Shows all calculations

# Get last result
last_result = calc.get_result()
```

## Examples

### Addition
```
Enter your choice (1-9): 1
Enter first number: 10
Enter second number: 5
Result: 15.0
```

### Division with Error Handling
```
Enter your choice (1-9): 4
Enter first number: 10
Enter second number: 0
Error: Cannot divide by zero
```

### View History
```
Enter your choice (1-9): 7
==================================================
CALCULATION HISTORY
==================================================
1. 10.0 + 5.0 = 15.0
2. 10.0 - 3.0 = 7.0
3. 2.0 * 8.0 = 16.0
==================================================
```

## Error Handling

The calculator includes proper error handling for:
- **Division by Zero**: Raises `ValueError` with message "Cannot divide by zero"
- **Negative Square Root**: Raises `ValueError` with message "Cannot calculate square root of negative number"
- **Invalid Input**: The CLI handles non-numeric input gracefully with user-friendly error messages

## Testing

The test suite includes 11 test cases that verify:
- Correct calculation results
- Proper error handling
- History tracking
- Edge cases (zero values, negative numbers, decimals)

All tests should pass when running the test suite.
