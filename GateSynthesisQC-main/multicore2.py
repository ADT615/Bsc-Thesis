import concurrent.futures
import math

it_parameters = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def it_function(parameters):
    if parameters < 2:
        return False
    if parameters == 2:
        return True
    if parameters % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(parameters)))
    for i in range(3, sqrt_n + 1, 2):
        if parameters % i == 0:
            return str(1) + 'xx'
    return 0

def main1(func, params):
    k = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number in (executor.map(func, params)):
            k.append((number))
    return k
import time
if __name__ == '__main__':
    start = time.time()
    k = main1(it_function, it_parameters)
    print(k)
    end = time.time()
    print('Time elapsed multi: ', end - start)
    start = time.time()
    for i in it_parameters:
        print(it_function(i))
    end = time.time()
    print('Time elapsed single: ', end - start)