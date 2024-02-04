def fibonacci(length, value0=0, value1=1): 
    if length == 2: 
        return value0+value1
    return fibonacci(length-1, value1, value0 + value1)

if __name__ == "__main__": 
    exercises = [6, 19, 28, 36, 38]
    for exercise in exercises: 
        print(fibonacci(exercise))
