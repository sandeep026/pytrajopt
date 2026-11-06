
import ast
import operator
import casadi as cs

# Symbolic variables
x = cs.SX.sym("x", 2)
u = cs.SX.sym("u")

# Allowed names
namespace = {
    "x": x,
    "u": u,
    "sin": cs.sin,
    "cos": cs.cos,
    "sqrt": cs.sqrt,
    "pi": cs.pi,
}

# Allowed operators
operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def compile_expr(expression):
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        if isinstance(node, ast.Name):
            if node.id not in namespace:
                raise ValueError(f"Unknown name: {node.id}")
            return namespace[node.id]

        if isinstance(node, ast.Subscript):
            return visit(node.value)[visit(node.slice)]

        if isinstance(node, ast.BinOp):
            op = operators.get(type(node.op))
            if op is None:
                raise ValueError("Operator not allowed")
            return op(visit(node.left), visit(node.right))

        if isinstance(node, ast.UnaryOp):
            op = operators.get(type(node.op))
            if op is None:
                raise ValueError("Unary operator not allowed")
            return op(visit(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are allowed")
            func = namespace.get(node.func.id)
            if not callable(func):
                raise ValueError(f"Unknown function: {node.func.id}")
            return func(*[visit(arg) for arg in node.args])

        raise ValueError(f"Expression not allowed: {type(node).__name__}")

    return visit(tree)


# Try it
expression = "x[0]**2 + x[1]**2 + sin(u)"

result = compile_expr(expression)

print("Symbolic expression:", result)

f = cs.Function("f", [x, u], [result])

print("Numerical result:", f([3, 4], 0.5))
