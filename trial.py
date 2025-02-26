def f(a, b):
    if a == b:
        return b
    if a + 2 == 1:
        return 2 * f((a-1)/2, b)
    return b + f(a-1, b)

print(f(15, 10))
