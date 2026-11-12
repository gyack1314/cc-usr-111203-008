"""小规模纯Python线性代数：Cholesky分解、求解、对数行列式与主元秩。

仅服务调和分析，不依赖第三方库；输入矩阵按列表的列表存放。
"""
import math


def cholesky(a):
    """返回下三角L使 a = L L^T；a必须对称正定，否则报错。"""
    n = len(a)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if not s > 0.0:
                    raise ValueError('矩阵非正定，无法Cholesky分解')
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def forward(L, b):
    """解下三角方程 L x = b。"""
    n = len(L)
    x = [0.0] * n
    for i in range(n):
        x[i] = (b[i] - sum(L[i][k] * x[k] for k in range(i))) / L[i][i]
    return x


def backward(L, b):
    """解上三角方程 L^T x = b，L为下三角。"""
    n = len(L)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i]
    return x


def chol_solve(L, b):
    """已知 a = L L^T，解 a x = b。"""
    return backward(L, forward(L, b))


def logdet(L):
    """由Cholesky因子求对数行列式 log|a| = 2*sum(log L_ii)。"""
    return 2.0 * sum(math.log(L[i][i]) for i in range(len(L)))


def inverse(a):
    """对称正定矩阵的逆，经Cholesky逐列求解。"""
    L = cholesky(a)
    n = len(a)
    cols = [chol_solve(L, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def matmul(a, b):
    inner = len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(len(b[0]))] for i in range(len(a))]


def matvec(a, x):
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in a]


def transpose(a):
    return [list(col) for col in zip(*a)]


def pivoted_rank(a, tol):
    """对称半定矩阵的主元Cholesky数值秩。

    每步取剩余对角最大者为主元，主元 <= tol*初始最大对角 时停止。
    返回 (秩, 被识别为冗余的下标列表)。
    """
    n = len(a)
    res = [a[i][i] for i in range(n)]
    base = max(res) if res else 0.0
    L = [[0.0] * n for _ in range(n)]
    remaining = set(range(n))
    order = []
    rank = 0
    for _ in range(n):
        p = max(remaining, key=lambda i: res[i])
        if res[p] <= tol * base:
            break
        order.append(p)
        remaining.discard(p)
        pivot = math.sqrt(res[p])
        L[p][rank] = pivot
        for i in remaining:
            L[i][rank] = (a[i][p] - sum(L[i][k] * L[p][k] for k in range(rank))) / pivot
            res[i] -= L[i][rank] ** 2
        rank += 1
    return rank, [i for i in range(n) if i not in order]
