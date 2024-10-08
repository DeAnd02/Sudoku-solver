import numpy as np
from scipy.optimize import minimize


def parse_board(board):
    return np.array([[int(c) if c != '.' else 0 for c in row] for row in board.split()])


def print_board(board):
    for row in board:
        print(' '.join(str(int(round(x))) if x != 0 else '.' for x in row))


def create_initial_x(board):
    x = np.zeros((9, 9, 9))
    for i in range(9):
        for j in range(9):
            if board[i, j] != 0:
                x[i, j, board[i, j] - 1] = 1
            else:
                x[i, j, :] = 1 / 9
    return x.flatten()


def objective(x):
    return 0


def constraint_sum_to_one(x):
    x = x.reshape((9, 9, 9))
    return np.sum(x, axis=2).flatten() - 1


def constraint_row(x):
    x = x.reshape((9, 9, 9))
    return np.sum(x, axis=1).flatten() - 1


def constraint_col(x):
    x = x.reshape((9, 9, 9))
    return np.sum(x, axis=0).flatten() - 1


def constraint_box(x):
    x = x.reshape((9, 9, 9))
    constraints = []
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = x[i:i + 3, j:j + 3, :]
            constraints.extend(np.sum(box, axis=(0, 1)) - 1)
    return np.array(constraints)


def constraint_fixed(x, board):
    x = x.reshape((9, 9, 9))
    constraints = []
    for i in range(9):
        for j in range(9):
            if board[i, j] != 0:
                constraints.append(x[i, j, board[i, j] - 1] - 1)
    return np.array(constraints)


def project_simplex(y):
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(len(y)) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(y - theta, 0)
    return w


def project(x):
    x = x.reshape((9, 9, 9))
    for i in range(9):
        for j in range(9):
            x[i, j] = project_simplex(x[i, j])
    return x.flatten()


def solve_sudoku(board):
    initial_x = create_initial_x(board)

    constraints = [
        {'type': 'eq', 'fun': constraint_sum_to_one},
        {'type': 'eq', 'fun': constraint_row},
        {'type': 'eq', 'fun': constraint_col},
        {'type': 'eq', 'fun': constraint_box},
        {'type': 'eq', 'fun': lambda x: constraint_fixed(x, board)}
    ]

    bounds = [(0, 1) for _ in range(9 * 9 * 9)]

    result = minimize(
        objective,
        initial_x,
        method='SLSQP',
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )

    solution = result.x.reshape((9, 9, 9))
    return np.argmax(solution, axis=2) + 1


# Esempio di utilizzo
sudoku_board = """
37. 5.. ..6
... 36. .12
... .91 75.
... 154 .7.
..3 .7. 6..
.5. 638 ...
.64 98. ...
59. .26 ...
2.. ..5 .64
"""

board = parse_board(sudoku_board)
print("Sudoku iniziale:")
print_board(board)

solved_board = solve_sudoku(board)
print("\nSoluzione:")
print_board(solved_board)