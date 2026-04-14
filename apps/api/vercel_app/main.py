"""
Main entry point for the Calculator CLI application
Provides an interactive command-line interface for the calculator
"""

from calculator import Calculator


def print_menu():
    """Display the calculator menu"""
    print("\n" + "=" * 50)
    print("         BASIC CALCULATOR APPLICATION")
    print("=" * 50)
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")
    print("5. Power")
    print("6. Square Root")
    print("7. View History")
    print("8. Clear History")
    print("9. Exit")
    print("=" * 50)


def get_two_numbers():
    """Get two numbers from the user"""
    while True:
        try:
            a = float(input("Enter first number: "))
            b = float(input("Enter second number: "))
            return a, b
        except ValueError:
            print("Invalid input. Please enter valid numbers.")


def get_one_number():
    """Get one number from the user"""
    while True:
        try:
            a = float(input("Enter a number: "))
            return a
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def main():
    """Main function to run the calculator application"""
    calc = Calculator()

    while True:
        print_menu()
        choice = input("Enter your choice (1-9): ")

        try:
            if choice == "1":
                a, b = get_two_numbers()
                result = calc.add(a, b)
                print(f"\nResult: {result}")

            elif choice == "2":
                a, b = get_two_numbers()
                result = calc.subtract(a, b)
                print(f"\nResult: {result}")

            elif choice == "3":
                a, b = get_two_numbers()
                result = calc.multiply(a, b)
                print(f"\nResult: {result}")

            elif choice == "4":
                a, b = get_two_numbers()
                result = calc.divide(a, b)
                print(f"\nResult: {result}")

            elif choice == "5":
                a, b = get_two_numbers()
                result = calc.power(a, b)
                print(f"\nResult: {result}")

            elif choice == "6":
                a = get_one_number()
                result = calc.square_root(a)
                print(f"\nResult: {result}")

            elif choice == "7":
                history = calc.get_history()
                if not history:
                    print("\nNo calculation history available.")
                else:
                    print("\n" + "=" * 50)
                    print("CALCULATION HISTORY")
                    print("=" * 50)
                    for idx, item in enumerate(history, 1):
                        print(f"{idx}. {item}")
                    print("=" * 50)

            elif choice == "8":
                calc.clear_history()
                print("\nHistory cleared!")

            elif choice == "9":
                print("\nThank you for using the calculator. Goodbye!")
                break

            else:
                print("\nInvalid choice. Please enter a number between 1 and 9.")

        except ValueError as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
