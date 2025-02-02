# def add(x, y, z):
#     return x + y + z
#
# sum = add(10, 20, 30)
# print(sum)


# bob_stats.py

def avg(nums):
    if len(nums) == 0:
        return 0
    s = sum(nums)
    c = len(nums)
    return s / c

def max_val(nums):
    if not nums:
        return None
    m = nums[0]
    for n in nums:
        if n > m:
            m = n
    return m

def main():
    data = [5, 15, 25, 35, 45]
    a = avg(data)
    m = max_val(data)
    print("Avg:", a)
    print("Max:", m)

if __name__ == "__main__":
    main()