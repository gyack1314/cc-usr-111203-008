"""按实际采样时刻的调和分析：GLS加谐波惩罚、候选评分、冻结外期。

只用真实观测时刻，不把行序号当时间，不对缺采时段插值，不预测未观测
过程；残差仅为观测减确定性拟合，不代表任何气候因素的因果贡献。
"""
import datetime
import math
from .io import number, positive
from .linalg import (cholesky, forward, chol_solve, logdet, inverse,
                     matmul, matvec, transpose, pivoted_rank)

LOG2PI = math.log(2.0 * math.pi)


def epoch(text):
    """epoch必须为固定的YYYY-MM-DDTHH:MM:SSZ形式，原样保留为元数据。"""
    if not isinstance(text, str):
        raise ValueError('epoch应为字符串')
    try:
        d = datetime.datetime.strptime(text, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        raise ValueError('epoch格式应为YYYY-MM-DDTHH:MM:SSZ') from None
    if d.strftime('%Y-%m-%dT%H:%M:%SZ') != text:
        raise ValueError('epoch格式必须固定')
    return text


def windows(spec):
    """三个含端点且互不交叠的时间窗。"""
    out = {}
    for name in ('early', 'validation', 'external'):
        lo, hi = spec[name]
        lo = number(lo)
        hi = number(hi)
        if lo > hi:
            raise ValueError('窗口端点次序不合法')
        out[name] = (lo, hi)
    items = list(out.items())
    for i in range(3):
        for j in range(i + 1, 3):
            (_, (a0, a1)), (_, (b0, b1)) = items[i], items[j]
            if a0 <= b1 and b0 <= a1:
                raise ValueError('窗口互相交叠')
    return out


def prepare(v):
    """校验资料并整理为升序唯一时刻序列，同刻同内容合并、冲突拒绝。"""
    ep = epoch(v['epoch'])
    bytime = {}
    for r in v['measurements']:
        t = number(r['hour'])
        y = number(r['level'])
        s = positive(r['sigma'])
        old = bytime.get(t)
        if old is not None:
            if old != (y, s):
                raise ValueError('同刻潮位资料冲突')
        else:
            bytime[t] = (y, s)
    if not bytime:
        raise ValueError('缺少潮位')
    times = sorted(bytime)
    omegas = [positive(w) for w in v['frequencies']]
    if len(set(omegas)) != len(omegas):
        raise ValueError('频率重复')
    ts = positive(v['time_scale'])
    wins = windows(v['windows'])
    belong = {}
    for t in times:
        hits = [name for name, (lo, hi) in wins.items() if lo <= t <= hi]
        if not hits:
            raise ValueError('时刻不在任何窗口')
        belong[t] = hits[0]
    for name in ('early', 'validation'):
        if not any(belong[t] == name for t in times):
            raise ValueError('窗口缺少资料:' + name)
    cands = []
    for c in v['candidates']:
        cid = c['id']
        if not isinstance(cid, str) or not cid:
            raise ValueError('候选id应非空')
        rho = number(c['rho'])
        ridge = number(c['ridge'])
        if not 0 <= rho < 1 or ridge < 0:
            raise ValueError('核或惩罚参数不合法')
        cands.append({'id': cid, 'rho': rho, 'ridge': ridge})
    if not cands:
        raise ValueError('缺少候选')
    ids = [c['id'] for c in cands]
    if len(set(ids)) != len(ids):
        raise ValueError('候选id重复')
    rank_tol = positive(v['rank_tolerance'])
    floor = number(v['phase_amplitude_floor'])
    if floor < 0:
        raise ValueError('相位振幅下限不合法')
    return {'epoch': ep, 'times': times,
            'level': [bytime[t][0] for t in times],
            'sigma': [bytime[t][1] for t in times],
            'window': [belong[t] for t in times],
            'omegas': omegas, 'time_scale': ts, 'candidates': cands,
            'rank_tol': rank_tol, 'phase_floor': floor}


def column_names(omegas):
    """设计列顺序：1, t, cos(omega1*t), sin(omega1*t), ..."""
    names = ['intercept', 'trend']
    for w in omegas:
        names += ['cos(omega=%r)' % w, 'sin(omega=%r)' % w]
    return names


def design(times, omegas):
    """设计矩阵，t为相对epoch的小时，禁止用行序号代替。"""
    X = []
    for t in times:
        row = [1.0, float(t)]
        for w in omegas:
            row += [math.cos(w * t), math.sin(w * t)]
        X.append(row)
    return X


def covariance(times, sig, rho, ts):
    """C_ij = s_i*s_j*rho^(|t_i-t_j|/ts)；rho=0时仅对角为s_i^2。"""
    n = len(times)
    C = [[0.0] * n for _ in range(n)]
    for i in range(n):
        C[i][i] = sig[i] * sig[i]
        if rho > 0.0:
            for j in range(i):
                c = sig[i] * sig[j] * rho ** (abs(times[i] - times[j]) / ts)
                C[i][j] = c
                C[j][i] = c
    return C


def cross_covariance(ta, sa, tb, sb, rho, ts):
    """两组不同记录之间的协方差；rho=0时不同记录协方差为零。"""
    if rho == 0.0:
        return [[0.0] * len(tb) for _ in ta]
    return [[sa[i] * sb[j] * rho ** (abs(ta[i] - tb[j]) / ts)
             for j in range(len(tb))] for i in range(len(ta))]


def fit(times, y, sig, omegas, rho, ts, ridge, rank_tol):
    """GLS加仅惩罚谐波系数平方和的拟合。

    白化：C = L L^T，Xw = L^{-1} X，yw = L^{-1} y；
    目标：||yw - Xw beta||^2 + ridge*sum(beta_harmonic^2)，截距与趋势不惩罚。
    A = B + ridge*P，B = Xw^T Xw；未惩罚块秩亏直接报错，ridge不能顶替。
    """
    X = design(times, omegas)
    p = 2 + 2 * len(omegas)
    n = len(times)
    C = covariance(times, sig, rho, ts)
    L = cholesky(C)
    Xwt = [forward(L, col) for col in transpose(X)]
    yw = forward(L, y)
    B = [[sum(Xwt[i][k] * Xwt[j][k] for k in range(n)) for j in range(p)] for i in range(p)]
    g = [sum(Xwt[i][k] * yw[k] for k in range(n)) for i in range(p)]
    rank_full, redundant = pivoted_rank(B, rank_tol)
    rank_unpen, _ = pivoted_rank([row[:2] for row in B[:2]], rank_tol)
    rank_harm, _ = pivoted_rank([row[2:] for row in B[2:]], rank_tol)
    if rank_unpen < 2:
        raise ValueError('截距/趋势不可识别：未惩罚部分秩亏，ridge不能替代')
    if rank_full < p and ridge == 0.0:
        raise ValueError('谐波部分秩亏且无惩罚，目标无唯一解')
    A = [row[:] for row in B]
    for i in range(2, p):
        A[i][i] += ridge
    La = cholesky(A)
    beta = chol_solve(La, g)
    Ainv = inverse(A)
    cov = matmul(matmul(Ainv, B), Ainv)
    resid = [yw[k] - sum(Xwt[i][k] * beta[i] for i in range(p)) for k in range(n)]
    objective = sum(r * r for r in resid) + ridge * sum(beta[i] ** 2 for i in range(2, p))
    return {'beta': beta, 'cov': cov, 'objective': objective,
            'logdet_C': logdet(L), 'n': n, 'X': X, 'Xwt': Xwt, 'B': B,
            'rank': {'full': rank_full, 'unpenalized': rank_unpen,
                     'harmonic': rank_harm, 'columns': p,
                     'tolerance': rank_tol, 'redundant': redundant}}


def conditional_kernel(rt, tt, st, tv, sv, rho, ts):
    """给定训练残差rt，验证残差的条件均值与条件协方差。

    均值 = C_vt C_tt^{-1} rt；协方差 = C_vv - C_vt C_tt^{-1} C_tv。
    """
    nt = len(tt)
    nv = len(tv)
    Lt = cholesky(covariance(tt, st, rho, ts))
    Cvv = covariance(tv, sv, rho, ts)
    Cvt = cross_covariance(tv, sv, tt, st, rho, ts)
    alpha = chol_solve(Lt, rt)
    m = matvec(Cvt, alpha)
    cols = [chol_solve(Lt, Cvt[i]) for i in range(nv)]
    S = [[Cvv[i][k] - sum(Cvt[i][j] * cols[k][j] for j in range(nt))
          for k in range(nv)] for i in range(nv)]
    return m, S


def gaussian_nll(r, S):
    """零均值高斯负对数密度：维数*log(2pi)、logdet与二次型各半。"""
    L = cholesky(S)
    alpha = chol_solve(L, r)
    return 0.5 * (len(r) * LOG2PI + logdet(L) + sum(a * b for a, b in zip(r, alpha)))


def summarize_harmonics(beta, cov, omegas, floor):
    """每个频率的振幅与相位；振幅不超过floor时相位不可识别。"""
    out = []
    for k, w in enumerate(omegas):
        i = 2 + 2 * k
        j = i + 1
        a = beta[i]
        b = beta[j]
        R = math.hypot(a, b)
        block = [[cov[i][i], cov[i][j]], [cov[j][i], cov[j][j]]]
        entry = {'omega': w, 'cos': a, 'sin': b, 'amplitude': R, 'coef_cov': block}
        if R <= floor:
            entry['phase_rad'] = None
            entry['phase_identifiable'] = False
            entry['amplitude_se'] = None
            entry['phase_se'] = None
        else:
            va, vb, cab = block[0][0], block[1][1], block[0][1]
            entry['phase_rad'] = math.atan2(b, a)
            entry['phase_identifiable'] = True
            entry['amplitude_se'] = math.sqrt(max(0.0, a * a * va + 2 * a * b * cab + b * b * vb)) / R
            entry['phase_se'] = math.sqrt(max(0.0, b * b * va - 2 * a * b * cab + a * a * vb)) / (R * R)
        out.append(entry)
    return out


def identifiability(Xwt, names, rank, n):
    """混叠、极近频率与数据不足的诊断；只报告，不修补。"""
    p = len(names)
    messages = []
    if n < p:
        messages.append('记录数%d少于设计列数%d，模型不可完全识别' % (n, p))
    if rank['full'] < p:
        lack = [names[i] for i in rank['redundant']]
        messages.append('白化设计秩%d小于列数%d，疑似混叠或极近频率，冗余列:%s'
                        % (rank['full'], p, ','.join(lack)))
    if rank['unpenalized'] < 2:
        messages.append('截距/趋势块秩亏，ridge不能掩盖未惩罚部分秩亏')
    pairs = []
    for i in range(2, p):
        for j in range(i + 1, p):
            ni = math.sqrt(sum(x * x for x in Xwt[i]))
            nj = math.sqrt(sum(x * x for x in Xwt[j]))
            if ni > 0 and nj > 0:
                c = sum(Xwt[i][k] * Xwt[j][k] for k in range(n)) / (ni * nj)
                if abs(c) > 0.9999:
                    pairs.append({'columns': [names[i], names[j]], 'correlation': c})
    if pairs:
        messages.append('存在近共线谐波列对，频率可能极近或混叠')
    return {'status': 'ok' if not messages else 'deficient',
            'messages': messages, 'near_collinear_pairs': pairs}


def shift_epoch(beta, cov, omegas, shift):
    """epoch平移shift小时（t_new = t_old - shift）时系数与协方差的一并变换。

    截距/趋势与每个谐波的cos/sin坐标同时旋转，不能只改数值时间而保持beta。
    """
    shift = number(shift)
    p = len(beta)
    M = [[0.0] * p for _ in range(p)]
    M[0][0] = 1.0
    M[0][1] = shift
    M[1][1] = 1.0
    for k, w in enumerate(omegas):
        th = w * shift
        c = math.cos(th)
        s = math.sin(th)
        i = 2 + 2 * k
        j = i + 1
        M[i][i] = c
        M[i][j] = s
        M[j][i] = -s
        M[j][j] = c
    return matvec(M, beta), matmul(matmul(M, cov), transpose(M))


def _score_on_validation(d, idx_t, idx_v, beta, cand):
    """代入早段beta的验证条件高斯负对数密度，不用验证点更新beta。"""
    times, y, sig = d['times'], d['level'], d['sigma']
    om, ts = d['omegas'], d['time_scale']
    tt = [times[k] for k in idx_t]
    st = [sig[k] for k in idx_t]
    tv = [times[k] for k in idx_v]
    sv = [sig[k] for k in idx_v]
    rt = [y[k] - sum(x * b for x, b in zip(design([times[k]], om)[0], beta)) for k in idx_t]
    rv = [y[k] - sum(x * b for x, b in zip(design([times[k]], om)[0], beta)) for k in idx_v]
    m, S = conditional_kernel(rt, tt, st, tv, sv, cand['rho'], ts)
    return gaussian_nll([rv[i] - m[i] for i in range(len(tv))], S)


def analyze(v):
    """完整流程：候选评分、开发段重拟合、冻结外期预测。"""
    d = prepare(v)
    times, y, sig = d['times'], d['level'], d['sigma']
    om, ts = d['omegas'], d['time_scale']
    idx = {name: [k for k, w in enumerate(d['window']) if w == name]
           for name in ('early', 'validation', 'external')}
    names = column_names(om)
    p = len(names)
    scores = []
    for cand in d['candidates']:
        entry = {'id': cand['id'], 'rho': cand['rho'], 'ridge': cand['ridge']}
        try:
            f = fit([times[k] for k in idx['early']], [y[k] for k in idx['early']],
                    [sig[k] for k in idx['early']], om, cand['rho'], ts,
                    cand['ridge'], d['rank_tol'])
            nll = _score_on_validation(d, idx['early'], idx['validation'], f['beta'], cand)
            if math.isfinite(nll):
                entry['score_nll'] = nll
                entry['status'] = 'ok'
            else:
                entry['score_nll'] = None
                entry['status'] = 'failed'
                entry['reason'] = '评分非有限值'
        except ValueError as e:
            entry['score_nll'] = None
            entry['status'] = 'failed'
            entry['reason'] = str(e)
        scores.append(entry)
    ok = [s for s in scores if s['score_nll'] is not None]
    if not ok:
        why = ';'.join('%s:%s' % (s['id'], s.get('reason', '?')) for s in scores)
        raise ValueError('所有候选拟合失败:' + why)
    best = min(ok, key=lambda s: (s['score_nll'], s['id']))
    cand = next(c for c in d['candidates'] if c['id'] == best['id'])
    dev = sorted(idx['early'] + idx['validation'], key=lambda k: times[k])
    td = [times[k] for k in dev]
    yd = [y[k] for k in dev]
    sd = [sig[k] for k in dev]
    fdev = fit(td, yd, sd, om, cand['rho'], ts, cand['ridge'], d['rank_tol'])
    beta = fdev['beta']
    cov = fdev['cov']
    Xd = fdev['X']
    series = []
    for i, k in enumerate(dev):
        trend = beta[0] + beta[1] * td[i]
        tidal = sum(Xd[i][j] * beta[j] for j in range(2, p))
        series.append({'hour': td[i], 'window': d['window'][k], 'level': yd[i],
                       'sigma': sd[i], 'trend': trend, 'tidal': tidal,
                       'residual': yd[i] - trend - tidal})
    pred_cov = matmul(matmul(Xd, cov), transpose(Xd))
    ext = []
    for k in idx['external']:
        row = design([times[k]], om)[0]
        pred = sum(x * b for x, b in zip(row, beta))
        ext.append({'hour': times[k], 'level': y[k], 'prediction': pred,
                    'residual': y[k] - pred})
    columns = [{'name': 'intercept', 'meaning': '截距：epoch时刻的确定性水位', 'penalized': False},
               {'name': 'trend', 'meaning': '线性趋势：每小时变化量', 'penalized': False}]
    for w in om:
        columns.append({'name': 'cos(omega=%r)' % w, 'meaning': '频率%r rad/小时的余弦分量' % w, 'penalized': True})
        columns.append({'name': 'sin(omega=%r)' % w, 'meaning': '频率%r rad/小时的正弦分量' % w, 'penalized': True})
    return {'epoch': d['epoch'], 'time_unit': 'hours_relative_to_epoch',
            'frequencies': om, 'time_scale': ts, 'columns': columns,
            'whitening': {'decomposition': 'C = L L^T（下三角Cholesky），白化设计 Xw = L^{-1} X',
                          'logdet_C_development': fdev['logdet_C']},
            'objective': {'formula': 'sum((L^{-1}(y - X beta))^2) + ridge * sum(beta_harmonic^2)',
                          'penalized': '仅谐波系数，截距与趋势不惩罚',
                          'ridge': cand['ridge'], 'value_development': fdev['objective']},
            'effective_rank': {'columns': p, 'full': fdev['rank']['full'],
                               'unpenalized': fdev['rank']['unpenalized'],
                               'harmonic': fdev['rank']['harmonic'],
                               'tolerance': fdev['rank']['tolerance'],
                               'redundant_columns': [names[i] for i in fdev['rank']['redundant']]},
            'identifiability': identifiability(fdev['Xwt'], names, fdev['rank'], fdev['n']),
            'candidates': scores,
            'selected': {'id': best['id'], 'score_nll': best['score_nll'],
                         'tie_break': '分数相同按候选id字典序取先'},
            'development': {
                'n': fdev['n'], 'beta_order': names,
                'beta': dict(zip(names, beta)), 'beta_covariance': cov,
                'covariance_note': 'A^{-1}(X^T C^{-1} X)A^{-1}三明治；ridge>0时A^{-1}本身不是抽样协方差；'
                                   '仅描述固定rho/ridge下测量噪声的抽样变化，不含候选选择不确定性',
                'harmonics': summarize_harmonics(beta, cov, om, d['phase_floor']),
                'series': series, 'prediction_covariance': pred_cov,
                'prediction_note': '跨时刻预测差方差 Var(p_i - p_j) = P_ii + P_jj - 2*P_ij，含共享系数协方差'},
            'external': {'frozen': True,
                         'note': '外期不参与beta估计与候选选择，仅给冻结预测与残差',
                         'series': ext},
            'residual_note': '残差 = 观测 - 确定性拟合；不对未观测时段插值或预测，'
                             '残差不归因于任何气候因素的因果贡献'}
