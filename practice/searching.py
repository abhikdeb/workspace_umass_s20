import numpy as np
import math
import time


# O(n)
def linear_search(arr, ele):
    pos = -1
    for i in range(len(arr)):
        if arr[i] == ele:
            pos = i
            return pos
    return pos


# O(log(n))
def binary_search(arr, ele, start=0):
    n = len(arr)
    mid_pos = int(n/2)

    if ele == arr[mid_pos]:
        return start + mid_pos
    elif arr[mid_pos] < ele:
        return binary_search(arr[mid_pos:], ele, start + mid_pos)
    else:
        return binary_search(arr[:mid_pos], ele, start)


# O(sqrt(n))
def jump_search(arr, ele):
    pos = 0
    n = len(arr)
    step = int(math.sqrt(len(arr)))
    block_pos = 0

    if arr[0] > ele or arr[n - 1] < ele:
        return -1

    while block_pos < n - step:
        if arr[block_pos] <= ele <= arr[block_pos + step]:
            break
        block_pos += step

    for i in range(block_pos, block_pos + step):
        if arr[i] == ele:
            pos = i

    return pos


def main():
    print("Executing...")
    arr = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    start = time.perf_counter_ns()
    ls = linear_search(arr, 55)
    end_ls = time.perf_counter_ns()
    print('Pos: ', ls, f' | Exec time: {end_ls - start:0.4f} nanoseconds')
    bs = binary_search(arr, 55)
    end_bs = time.perf_counter_ns()
    print('Pos: ', bs, f' | Exec time: {end_bs - end_ls:0.4f} nanoseconds')
    js = jump_search(arr, 55)
    end_js = time.perf_counter_ns()
    print('Pos: ', js, f' | Exec time: {end_js - end_bs:0.4f} nanoseconds')
    exit(0)
    return True


def max_subarray(arr):
    max_sum = arr[0]
    curr_sum = arr[0]
    for i in range(len(arr)):
        curr_sum = max(arr[i], curr_sum + arr[i])
        max_sum = max(max_sum, curr_sum)
    return max_sum


def main_1():
    arr = [1, 2, 3, 4, 4, -5, 56, -67, 3]
    n = len(arr)
    k = 3

    # for i in range(n):
    #     for j in range(i-k, i+k+1):
    #         if 0 <= j < n:
    #             print(i, '|', j)

    print(max_subarray(arr))

    return


if __name__ == "__main__":
    # main()
    main_1()
