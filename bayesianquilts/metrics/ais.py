#!/usr/bin/env python3

from abc import ABC, abstractmethod


class AdaptiveIsSampler(ABC):
    def __init__(self ):
        return
    
class Bijection(ABC):
    @abstractmethod
    def call(data, **params):
        return
    
    @abstractmethod
    def inverse():
        return 
    
    @abstractmethod
    def forward_grad():
        return
    
    def __init__(self):
        return
    
class AutoDiffBijection(Bijection):
    def __init__(self, model, hbar=1.0):
        self.model = model
        self.hbar = hbar
        return

    def call(self, data, params):
        return self.model.adaptive_is_loo(data, params, self.hbar)
    
class SmallStepTransformation(Bijection):
    def call():
        return