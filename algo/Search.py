class Search:

    def sequential_search(self,a, x):
        n = len(a)
        for i in range(0,n):
            if a[i] == x:
                return i
        return -1