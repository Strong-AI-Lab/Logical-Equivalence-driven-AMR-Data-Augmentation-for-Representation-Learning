list1 = [0, 50, 80, 130, 180, 280, 300]
k= 335
flag = len(list1)-1
return_list = []
while flag != 0:
    initial = 0
    gap = k - list1[flag]
    for item in range(0,flag):
        if list1[item] > gap and (list1[item],list1[flag]) not in return_list and (list1[flag], list1[item]) not in return_list:
            return_list.append((list1[item],list1[flag]))
    flag = flag - 1
print(len(return_list))