# from https://python.plainenglish.io/five-python-wrappers-that-can-reduce-your-code-by-half-af775feb1d5

import time

def timer(func):
    def wrapper(*args, **kwargs):
        # start the timer
        start_time = time.time()
        # call the decorated function
        result = func(*args, **kwargs)
        # remeasure the time
        end_time = time.time()
        # compute the elapsed time and print it
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds, function: {func.__name__} with args: {args} kwargs: {kwargs}")
        # return the result of the decorated function execution
        return result
    # return reference to the wrapper function
    return wrapper

#usage
# @timer
# def train_model():
#     print("Starting the model training function...")
#     # simulate a function execution by pausing the program for 5 seconds
#     time.sleep(5) 
#     print("Model training completed!")

def debug(func):
    def wrapper(*args, **kwargs):
        # print the fucntion name and arguments
        print(f"Calling {func.__name__} with args: {args} kwargs: {kwargs}")
        # call the function
        result = func(*args, **kwargs)
        # print the results
        print(f"{func.__name__} returned: {result}")
        return result
    return wrapper

# @debug
# def add_numbers(x, y):
#     return x + y
# add_numbers(7, y=5,)  # Output: Calling add_numbers with args: (7) kwargs: {'y': 5} \n add_numbers returned: 12

def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Handle the exception
            print(f"An exception occurred: {str(e)}")
            # Optionally, perform additional error handling or logging
            # Reraise the exception if needed
    return wrapper

# @exception_handler
# def divide(x, y):
#     result = x / y
#     return result
# divide(10, 0)  # Output: An exception occurred: division by zero

def validate_input(*validations):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i, val in enumerate(args):
                if i < len(validations):
                    if not validations[i](val):
                        raise ValueError(f"Invalid argument: {val}")
            for key, val in kwargs.items():
                if key in validations[len(args):]:
                    if not validations[len(args):][key](val):
                        raise ValueError(f"Invalid argument: {key}={val}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# @validate_input(lambda x: x > 0, lambda y: isinstance(y, str))
# def divide_and_print(x, message):
#     print(message)
#     return 1 / x

# divide_and_print(5, "Hello!")  # Output: Hello! 1.0

