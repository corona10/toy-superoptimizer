## Toy Superoptimizer
Toy superoptimizer which is inspired by https://austinhenley.com/blog/superoptimizer.html
Only supports a few OPCODES of CPython 3.11, needs to optimize search space and searching performance.
And also need to add more opcodes that more python code can be executed.

### code
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
