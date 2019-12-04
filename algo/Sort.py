class Sort:

    def select_sort(self,v):
        n = len(v)
        for i in range(0, n-1):
            min_idx = i
            for j in range(i+1, n):
                if v[j] < v[min_idx]:
                    min_idx = j
                print('i/j : ' + str(i) + '/' + str(j) + ', min_idx : ' + str(min_idx))
            v[i],  v[min_idx] = v[min_idx], v[i]
        return v

    def insert_sort(self,v):
        n = len(v)
        for i in range(1, n):
            key = v[i]
            j = i - 1
            while j >= 0 and v[j] > key :
                v[j + 1] = v[j]
                j -= 1
            v[j+1] = key
        return v