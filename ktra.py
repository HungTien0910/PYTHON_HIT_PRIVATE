n = int(input())
a= list(map(int, input().split()))
b= [i for i in a if sum([j for j in range(1,i) if i % j ==0 ]) > i]
print(len(b))
print(sorted(b))

a = input().split()
dem =0
for i in range(len(a[1])):
    if a[0] == a[1][i:i+2]:
        dem +=1
print(dem)
# 
n = int(input())
dem = 0 
b = []
a = list(map(int, input().split()))
for i in a:
    for j in range(1,i+1):
        if i % j ==0 :
            dem +=1
            if dem ==3 :
                b.append(i)
if len(b) == 0:
    print("KHÔNG") 
else:
    print(len(b))   

a= list(map(str,input().split()))
n = int(input())
b = list(set(i for i in a[0]))



