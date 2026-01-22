def test():
    return (1, 2, 3), (4, 5, 6)


(a, b, c), (x, y, z) = test()

print(a, b, c, x, y, z)