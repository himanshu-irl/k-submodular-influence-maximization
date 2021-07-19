# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 22:15:18 2021

@author: hverm
"""

def _func(s, i):
        # return 1812433253 * (s ^ (s >> 30)) + i + 1;
        return 1812433253 * (s ^ (s >> 30)) + i + 1
    
class Xorshift:
    def __init__(self, seed):
        self.seed = seed
        self.x = _func(self.seed,0)
        self.y = _func(self.x,1)
        self.z = _func(self.y,2)
        self.w = _func(self.z,3)
    
    # def Xorshift(self):
    # x = _(seed,0)
    # y = _(x,1)
    # z = _(y,2)
    # w = _(z,3)
        
    # def _func(s, i):
    #     # return 1812433253 * (s ^ (s >> 30)) + i + 1;
    #     return 1812433253 * (s ^ (s >> 30)) + i + 1
    
    def gen_int(self):
        t = self.x ^ (self.x << 11)
        self.x = self.y
        self.y = self.z
        self.z = self.w
        self.w = self.w ^ (self.w >> 19) ^ t ^ (t >> 8)
        return self.w
    
    def nextInt(n):
        return int(n * nextDouble())
    
    def nextDouble(self):
        a = self.gen_int() >> 5
        b = self.gen_int() >> 6
        return (a * 67108864.0 + b) / (1 << 53)


z = Xorshift(0)
z.gen_int()

z.nextDouble()

z.__dict__


z.x
z.y
z.z
z.w