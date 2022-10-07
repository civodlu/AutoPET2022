import unittest
import auto_pet


class TestFunctionToTest(unittest.TestCase):
    def test_function_to_test(self):
        r = auto_pet.todo_function_to_test2(1, 2)
        print('Result=', r)
        assert r == 3

if __name__ == '__main__':
    unittest.main()