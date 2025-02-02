# def add(a, b):
#     return a + b
#
# result = add(5, 3)
# print(result)

# alice_calculations.py

def calculate_average(numbers):
    """
    Calculates the average of a list of numbers.

    Parameters:
        numbers (list of float): The numbers to calculate the average for.

    Returns:
        float: The average of the numbers.
    """
    if not numbers:
        return 0.0
    total_sum = sum(numbers)
    count = len(numbers)
    average = total_sum / count
    return average

def find_maximum(numbers):
    """
    Finds the maximum number in a list.

    Parameters:
        numbers (list of float): The list of numbers to search.

    Returns:
        float: The maximum number in the list.
    """
    if not numbers:
        return None
    maximum = numbers[0]
    for number in numbers:
        if number > maximum:
            maximum = number
    return maximum

def main():
    """
    Main function to demonstrate calculation of average and maximum.
    """
    data = [10.5, 23.3, 45.2, 67.1, 89.0]
    avg = calculate_average(data)
    max_num = find_maximum(data)
    print(f"The average is: {avg}")
    print(f"The maximum number is: {max_num}")

if __name__ == "__main__":
    main()