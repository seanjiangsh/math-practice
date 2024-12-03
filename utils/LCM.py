def calculate_lcm(nums):

    def lcm(x, y):
        if x > y:
            greater = x
        else:
            greater = y

        while True:
            if greater % x == 0 and greater % y == 0:
                return greater
            greater += 1

    num1 = nums[0]
    for num2 in nums[1:]:
        num1 = lcm(num1, num2)

    return num1


numbers = []
while True:
    num = input("Enter a number (or 'stop' to finish): ")
    if num == 'stop':
        break
    numbers.append(int(num))

print("The L.C.M. is", calculate_lcm(numbers))
