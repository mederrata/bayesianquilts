#!/usr/bin/env python3

from abc import ABC, abstractmethod


class AdaptiveIsSampler(ABC):
    def __init__(self ):
        return
    
class Bijection(ABC):
    @abstractmethod
    def call():
        return
    
class SmallStepTransformation(Bijection):
    def call():
        return