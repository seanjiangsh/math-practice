def get_prime_factors(num):
    factors = []
    divisor = 2

    while divisor <= num:
        if num % divisor == 0:
            factors.append(divisor)
            num = num / divisor
        else:
            divisor += 1

    return factors


# Get input from the user
num = int(input("Enter a number: "))

# Get the prime factors
prime_factors = get_prime_factors(num)

# Print the prime factors
print("Prime factors of", num, "are:", prime_factors)
