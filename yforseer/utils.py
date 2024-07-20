
def print_title(title: str, width: int = 80):
    print("=" * width)
    print(title.center(width))
    print("=" * width)


calc_kernel = lambda x, k, s:  (x-k)//s + 1
