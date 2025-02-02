# def calculate(num1, num2, num3):
#     return num1 + num2 * num3
#
# total = calculate(7, 8, 9)
# print(total)

# charlie_analysis.py

from functools import wraps

def debug(func):
    """
    Decorator to print function name and arguments.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_list = ', '.join(map(str, args))
        kwargs_list = ', '.join(f"{k}={v}" for k, v in kwargs.items())
        print(f"Calling {func.__name__}({args_list}, {kwargs_list})")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@debug
def compute_statistics(data):
    """
    Computes statistics using list comprehensions and lambda functions.

    Parameters:
        data (list of float): The dataset.

    Returns:
        dict: Dictionary containing mean, median, and variance.
    """
    mean = sum(data) / len(data) if data else 0
    sorted_data = sorted(data)
    mid = len(data) // 2
    median = sorted_data[mid] if len(data) % 2 != 0 else (sorted_data[mid - 1] + sorted_data[mid]) / 2
    variance = sum((x - mean) ** 2 for x in data) / len(data) if data else 0
    return {'mean': mean, 'median': median, 'variance': variance}

def main():
    dataset = [12, 7, 3, 14, 9, 11, 8]
    stats = compute_statistics(dataset)
    print(f"Statistics: {stats}")

if __name__ == "__main__":
    main()