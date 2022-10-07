from unittest import TestCase
import trw


class TestOrderedSet(TestCase):
    def test_order_from_constructor(self):
        i = [-3, 1, 4, -1, 8]
        s = trw.simple_layers.OrderedSet(i)
        i_output = list(s)
        assert i == i_output
        assert len(s) == len(i)
        
        assert -3 in s
        
        s.discard(8)
        assert len(s) == len(i) - 1

        assert 8 not in s
        
    def test_reversed(self):
        i = [-3, 1, 4, -1, 8]
        s = trw.simple_layers.OrderedSet(i)

        assert list(reversed(s)) == list(reversed(i))
        
    def test_equal(self):
        i = [-3, 1, 4, -1, 8]
        s1 = trw.simple_layers.OrderedSet(i)
        s2 = trw.simple_layers.OrderedSet(i)
        assert s1 == s2
        
        last = s2.pop()
        assert last == 8
        assert s1 != s2
        
        string = str(s2)
        assert string == 'OrderedSet([-3, 1, 4, -1])'
        
    def test_order_from_add(self):
        i = [-3, 1, 4, -1, 8]
        s = trw.simple_layers.OrderedSet()
        for value in i:
            s.add(value)
            
        i_output = list(s)
        assert i == i_output
