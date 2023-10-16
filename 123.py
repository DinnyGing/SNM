import random

x = 0
y = 0
# Sorts array a[0..n-1] using Bogo sort
def bogoSort(a):
    n = len(a)
    global x
    x = 0
    while not sorted(a) == a:
        random.shuffle(a)
        x += 1
    print(f"X: {x}")
    return x


# Driver code to test above

for _ in range(10):
    a = [3, 2, 4, 1, 0, 5, 6, 7, 8, 9, 10]
    y += bogoSort(a)
y /= 10
print(f"sum: {y}")
print("Sorted array:")
for i in range(len(a)):
    print ("%d" %a[i]),
