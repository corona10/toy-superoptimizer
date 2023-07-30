## Toy Superoptimizer
Toy superoptimizer which is inspired by https://austinhenley.com/blog/superoptimizer.html.
It only supports a subset of CPython 3.11 opcodes, and it also needs to optimize searching space and algorithm.
More efforts are needed to able to execute more Python codes.

** Only available in CPython 3.11 **

### Code
````python
def f():
    a, b = 3, 5
    a, b = b, a
    return a

program = Program.from_function(f)
optimizer = Superoptimizer(program)
optimizer.search()
````

### AS-IS
````
LOAD_CONST 1
UNPACK_SEQUENCE 2
STORE_FAST 0
STORE_FAST 1
LOAD_FAST 1
LOAD_FAST 0
STORE_FAST 1
STORE_FAST 0
LOAD_FAST 0
RETURN_VALUE
````

### TO-BE
````
LOAD_CONST 1
UNPACK_SEQUENCE 2
STORE_FAST 1
STORE_FAST 0
LOAD_FAST 0
RETURN_VALUE
````
