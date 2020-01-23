from algo.Search import Search
from algo.Sort import Sort
from algo.Iteration import Iteration

if __name__=="__main__":
    print('순차 탐색')
    search = Search()
    v = [17, 92, 18, 33, 7, 33, 42]
    f = 18
    print(search.sequential_search(v, f))

    print('선택 정렬')
    sort = Sort()
    v = [2, 4, 5, 1, 3]
    print(sort.select_sort(v))

    print('삽입 정렬')
    sort = Sort()
    v = [2, 4, 5, 1, 3]
    print(sort.select_sort(v))

    #Iterations
    # 양의정수를 받아서 바이너리열로 변환 했을 때, 가장 긴 1 사이의 갭을 구해라.
    iteration = Iteration()
    iteration.binary_gap(3012)









