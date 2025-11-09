import concurrent.futures

def bypass_compile(i):
    sum = 0
    for j in range(i, i + 500):
        sum += j
    return sum


def multiple_compile(arr):
    executor = concurrent.futures.ProcessPoolExecutor(16)
    results = executor.map(bypass_compile, arr)
    return results

if __name__ == '__main__':
    import time
    start = time.time()
    k = list(multiple_compile(range(100000)))
    end = time.time()
    print('Time elapsed multi: ', end - start)
    start = time.time()
    k = list(map(bypass_compile, range(100000)))
    end = time.time()
    print('Time elapsed single: ', end - start)
