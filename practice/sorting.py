

def sel_sort(arr):
    sorted_arr = []
    for i in range(len(arr)):
        min_val = min(arr)
        min_idx = arr.index(min(arr))
        sorted_arr.append(min_val)
        del arr[min_idx]
    return sorted_arr


def main():
    arr = [2,14,1,55,22,9,64,32,11,5,1,54,23,25,67,15,10,21]
    print(sel_sort(arr))
    exit()
    return True


if __name__ == "__main__":
    main()

