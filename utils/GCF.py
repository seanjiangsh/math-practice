import math


def calculate_gcf(nums):
    num1 = nums[0]
    for num2 in nums[1:]:
        num1 = math.gcd(num1, num2)

    return num1


numbers = []
while True:
    num = input("Enter a number (or 'stop' to finish): ")
    if num == 'stop':
        break
    numbers.append(int(num))

print("The G.C.F. is", calculate_gcf(numbers))
