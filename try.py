def foo(a, n=100):
    check = {}
    for i in range(n):
        print(a[int((i) * len(a) / n)])
        if a[int((i) * len(a) / n)] in check:
            check[a[int((i) * len(a) / n)]] += 1
        else:
            check[a[int((i) * len(a) / n)]] = 1
    return check



a = list(range(2, 26))


test = foo(a, n=100)
for k in a:
    print(k, '-', test[k])
