OPS = ['push','pop','top','pass']
SYM = ['a','b','$']

class Stack:    

    def __init__(self):
        self.v = []
    
    def push(self,x):
        self.v.append(x)
    
    def pop(self):
        if len(self.v) == 0:
            return '$'
        x = self.v[-1]
        self.v = self.v[:-1]
        return x
    
    def top(self):
        if len(self.v) == 0:
            return '$'
        return self.v[-1]

class BlackBox:
    def __init__(self):
        self.s = Stack()
        self.state = [None,'$']
    
    def read(self,in_c):
        self.state = [in_c,self.s.top()]
    
    def act(self,a,in_c):
        rval = None
        #print self.state,'__',
        if a == 'push':
            self.s.push(in_c)
        elif a == 'pop':
            rval = self.s.pop()
        elif a == 'top':
            rval = self.s.top()
        elif a == 'pass':
            rval = in_c
        self.state = [in_c,self.s.top()]
        #print self.state
        return rval
