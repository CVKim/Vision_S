class A:
    def __init__(self):
        print("This is class A")
        
        
        
        self.a = 1
        self.ab = 'ab'
        
    def a_function_1(self):
        print("This is a_function_1 from class A")
        
    def a_function_2(self, val):
        print("This is a_function_2 from class A: value is ", val)
        
        
    
        
        
class B:
    def __init__(self):
        print("This is class B")
        
        self.b = 1111
        self.bc = 'bc'
        
    def b_function_1(self):
        print("This is a_function_1 from class B")
        
    def b_function_2(self, val):
        print("This is a_function_2 from class B: value is ", val)
        
        
        
a = A()
a.a_function_1()
a.a_function_2(100)
b = B()
b.b_function_1()
b.b_function_2(200)


    