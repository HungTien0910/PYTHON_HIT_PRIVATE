import numpy as np
class Matrix:
    def __init__(self, n=0, m=0):
        self.n = n
        self.m = m
    def nhap(self):
        self.n = int(input())
        self.m = int(input())
        self.a = np.array(list(map(int, input().split()))).reshape(self.n, self.m)
        self.b = np.array(list(map(int, input().split()))).reshape(self.n, self.m)
    def xuat(self):
        print("Hai mang vua nhap: ")
        print(self.a)
        print(self.b)
        print("Phep cong: ")
        print(self.a + self.b)
        print("Phep tru: ")
        print(self.a - self.b)
        print("Phep nhan: ")
        print(self.a * self.b)
        print("Phep chia: ")
        print(self.a / self.b)
    # Câu 2:

    def nhapcau2(self):
        self.c = np.array([[int(i) for i in input() if i.isnumeric() == True] for j in range(self.n)])
        self.d = np.array([[int(i) for i in input() if i.isnumeric() == True] for j in range(self.n)])

    def xuat_cau2(self):
        print("Cau 2: ")
        self.t = self.c + self.d
        self.h = self.c - self.d
        self.mul = self.c * self.d
        self.div = self.c / self.d
        print("Phep cong: ")
        print(self.t)
        print("Phep tru: ")
        print(self.h)
        print("Phep nhan: ")
        print(self.mul)
        print("Phep chia: ")
        print(self.div)
    # Câu 3: 
    def xuat_cau3(self):
        print("Cau 3: ")
        print("Phep cong: ")
        for i in self.t:
            print('\t'.join(map(str, i)))
        print("Phep tru: ")
        for i in self.h:
            print('\t'.join(map(str, i)))
        print("Phep nhan: ")
        for i in self.mul:
            print('\t'.join(map(str, i)))
        print("Phep chia: ")
        for i in self.div:
            print('\t'.join(map(str, i)))
        
a = Matrix()
a.nhap()
a.xuat()
a.nhapcau2()
a.xuat_cau2()
a.xuat_cau3()
