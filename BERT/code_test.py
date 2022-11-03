import copy
l1 = [1,2,3,4,5]
# l2 = [6,7,8,9,10]
# l2 = [-2,-1,0]
l2 = [-1,0,1.5,2.5,3.5]

# if l1[len(l1)-1] <= l2[0]:
#     l1.extend(l2)
#     l = copy.deepcopy(l1)
# elif l1[0] >= l2[len(l2)-1]:
#     l2.extend(l1)
#     l = copy.deepcopy(l2)
# else:
total_list = []
length1 = len(l1)
length2 = len(l2)
i = 0
j = 0
while i!=length1 and j !=length2:
    if l1[i] < l2[j]:
        total_list.append(l1[i])
        i = i + 1
    elif l1[i] >= l2[j]:
        total_list.append(l2[j])
        j = j + 1
l = total_list
print(l)