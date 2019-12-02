from algo.Search import Search

if __name__=="__main__":
    print('순차 탐색')
    search = Search()
    v = [17, 92, 18, 33, 7, 33, 42]
    f = 18
    print(search.sequential_search(v, f))


