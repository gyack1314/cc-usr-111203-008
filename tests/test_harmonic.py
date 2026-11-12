"""小样本核对完整C、条件核与系数协方差；演示缺采、跨epoch与rho=0。"""
import copy
import json
import math
import random
import unittest
from climate_data.io import load
from climate_data.harmonic import (prepare, design, covariance, cross_covariance,
                                   fit, conditional_kernel, gaussian_nll,
                                   summarize_harmonics, shift_epoch, analyze)
from climate_data.linalg import (cholesky, forward, chol_solve, logdet,
                                 matmul, matvec, transpose)


def gj_inverse(a):
    """测试侧独立实现：Gauss-Jordan求逆，用于交叉核对。"""
    n = len(a)
    m = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(a)]
    for col in range(n):
        p = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[p] = m[p], m[col]
        piv = m[col][col]
        m[col] = [x / piv for x in m[col]]
        for r in range(n):
            if r != col:
                f = m[r][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [row[n:] for row in m]


def dataset(times, sig=0.2, omegas=(0.5,), candidates=None,
            wins=((-50, 0), (1, 50), (51, 100)), ts=1.0, level_fn=None):
    if level_fn is None:
        def level_fn(t):
            return 2.0 + 0.01 * t + 0.4 * math.cos(0.5 * t) - 0.3 * math.sin(0.5 * t)
    if candidates is None:
        candidates = [{'id': 'c0', 'rho': 0.0, 'ridge': 0.0}]
    ms = [{'hour': t, 'level': level_fn(t),
           'sigma': sig(t) if callable(sig) else sig} for t in times]
    return {'epoch': '2020-01-01T00:00:00Z', 'measurements': ms,
            'frequencies': list(omegas), 'time_scale': ts, 'candidates': candidates,
            'windows': {'early': list(wins[0]), 'validation': list(wins[1]),
                        'external': list(wins[2])},
            'rank_tolerance': 1e-10, 'phase_amplitude_floor': 1e-8}


class CovarianceTests(unittest.TestCase):
    def test_full_c_matches_formula_and_cholesky(self):
        t = [-1.0, 0.5, 3.0]
        s = [0.2, 0.1, 0.3]
        rho, ts = 0.5, 2.0
        C = covariance(t, s, rho, ts)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(C[i][j], s[i] * s[j] * rho ** (abs(t[i] - t[j]) / ts))
        L = cholesky(C)
        rec = matmul(L, transpose(L))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(rec[i][j], C[i][j], places=12)
        det = (C[0][0] * (C[1][1] * C[2][2] - C[1][2] * C[2][1])
               - C[0][1] * (C[1][0] * C[2][2] - C[1][2] * C[2][0])
               + C[0][2] * (C[1][0] * C[2][1] - C[1][1] * C[2][0]))
        self.assertAlmostEqual(math.exp(logdet(L)), det, places=12)

    def test_rho_zero_diagonal_only(self):
        C = covariance([-1.0, 0.5, 3.0], [0.2, 0.1, 0.3], 0.0, 2.0)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(C[i][j], [0.04, 0.01, 0.09][i] if i == j else 0.0)
        cross = cross_covariance([0.0], [0.2], [0.0, 1.0], [0.2, 0.1], 0.0, 2.0)
        self.assertEqual(cross, [[0.0, 0.0]])


class ConditionalTests(unittest.TestCase):
    def setUp(self):
        self.tt = [0.0, 2.5]
        self.st = [0.2, 0.15]
        self.tv = [1.0, 4.0, 7.5]
        self.sv = [0.1, 0.3, 0.2]
        self.rho = 0.6
        self.ts = 1.3

    def test_kernel_matches_direct_inverse(self):
        rt = [0.3, -0.2]
        m, S = conditional_kernel(rt, self.tt, self.st, self.tv, self.sv, self.rho, self.ts)
        Ctt = covariance(self.tt, self.st, self.rho, self.ts)
        Cvv = covariance(self.tv, self.sv, self.rho, self.ts)
        Cvt = cross_covariance(self.tv, self.sv, self.tt, self.st, self.rho, self.ts)
        W = matmul(Cvt, gj_inverse(Ctt))
        m2 = matvec(W, rt)
        for a, b in zip(m, m2):
            self.assertAlmostEqual(a, b, places=12)
        for i in range(3):
            for k in range(3):
                s2 = Cvv[i][k] - sum(W[i][j] * Cvt[k][j] for j in range(2))
                self.assertAlmostEqual(S[i][k], s2, places=12)

    def test_conditional_density_factorizes_joint(self):
        om = [0.7]
        beta = [2.0, 0.05, 0.4, -0.3]
        yt = [2.1, 1.7]
        yv = [2.3, 1.9, 2.5]
        rt = [y - sum(x * b for x, b in zip(design([t], om)[0], beta)) for t, y in zip(self.tt, yt)]
        rv = [y - sum(x * b for x, b in zip(design([t], om)[0], beta)) for t, y in zip(self.tv, yv)]
        m, S = conditional_kernel(rt, self.tt, self.st, self.tv, self.sv, self.rho, self.ts)
        nll_cond = gaussian_nll([r - mi for r, mi in zip(rv, m)], S)
        tall = self.tt + self.tv
        sall = self.st + self.sv
        nll_joint = gaussian_nll(rt + rv, covariance(tall, sall, self.rho, self.ts))
        nll_marg = gaussian_nll(rt, covariance(self.tt, self.st, self.rho, self.ts))
        self.assertAlmostEqual(nll_cond, nll_joint - nll_marg, places=10)


class MonteCarloTests(unittest.TestCase):
    def test_beta_covariance_matches_simulation(self):
        times = [0.0, 1.0, 2.5, 4.0, 6.0, 9.0, 10.0, 13.0]
        sig = [0.15, 0.20, 0.15, 0.25, 0.20, 0.15, 0.20, 0.25]
        om = [0.5]
        ts = 1.0
        rho = 0.6
        beta_true = [2.0, 0.01, 0.4, -0.3]
        X = design(times, om)
        C = covariance(times, sig, rho, ts)
        L = cholesky(C)
        mean_y = [sum(x * b for x, b in zip(row, beta_true)) for row in X]
        Xwt = [forward(L, col) for col in transpose(X)]
        rng = random.Random(20261112)
        reps = 2000
        for ridge in (0.0, 0.5):
            f = fit(times, [0.0] * 8, sig, om, rho, ts, ridge, 1e-12)
            B = f['B']
            A = [row[:] for row in B]
            for i in range(2, 4):
                A[i][i] += ridge
            La = cholesky(A)
            s1 = [0.0] * 4
            s2 = [[0.0] * 4 for _ in range(4)]
            for _ in range(reps):
                z = [rng.gauss(0.0, 1.0) for _ in times]
                y = [m + e for m, e in zip(mean_y, matvec(L, z))]
                yw = forward(L, y)
                g = [sum(Xwt[i][k] * yw[k] for k in range(8)) for i in range(4)]
                bh = chol_solve(La, g)
                for i in range(4):
                    s1[i] += bh[i]
                    for j in range(4):
                        s2[i][j] += bh[i] * bh[j]
            mean = [v / reps for v in s1]
            emp = [[s2[i][j] / reps - mean[i] * mean[j] for j in range(4)] for i in range(4)]
            the = f['cov']
            for i in range(4):
                self.assertAlmostEqual(emp[i][i], the[i][i], delta=0.2 * the[i][i])
                if ridge == 0.0:
                    self.assertAlmostEqual(mean[i], beta_true[i],
                                           delta=5 * math.sqrt(the[i][i] / reps))
            for i in range(4):
                for j in range(4):
                    ce = emp[i][j] / math.sqrt(emp[i][i] * emp[j][j])
                    ct = the[i][j] / math.sqrt(the[i][i] * the[j][j])
                    self.assertAlmostEqual(ce, ct, delta=0.1)
            if ridge == 0.0:
                Binv = gj_inverse(B)
                for i in range(4):
                    for j in range(4):
                        self.assertAlmostEqual(the[i][j], Binv[i][j], places=10)


class EpochTests(unittest.TestCase):
    def test_shift_epoch_matches_refit(self):
        times = [-2.0, 0.5, 3.0, 7.0]
        sig = [0.2, 0.15, 0.1, 0.25]
        om = [0.5, 1.1]
        y = [2.1, 1.8, 2.4, 1.6]
        f = fit(times, y, sig, om, 0.4, 2.0, 0.3, 1e-12)
        shift = 7.25
        nb, nc = shift_epoch(f['beta'], f['cov'], om, shift)
        f2 = fit([t - shift for t in times], y, sig, om, 0.4, 2.0, 0.3, 1e-12)
        for a, b in zip(nb, f2['beta']):
            self.assertAlmostEqual(a, b, delta=1e-8 * max(1.0, abs(b)))
        for r1, r2 in zip(nc, f2['cov']):
            for a, b in zip(r1, r2):
                self.assertAlmostEqual(a, b, delta=1e-8 * max(1.0, abs(b)))
        for t in times:
            x1 = design([t], om)[0]
            x2 = design([t - shift], om)[0]
            p1 = sum(x * b for x, b in zip(x1, f['beta']))
            p2 = sum(x * b for x, b in zip(x2, nb))
            self.assertAlmostEqual(p1, p2, delta=1e-9)

    def test_cross_epoch_data_fit(self):
        v = dataset([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 30.0, 31.0, 60.0],
                    wins=((-5, 0), (1, 40), (41, 80)))
        rep = analyze(v)
        hours = [r['hour'] for r in rep['development']['series']]
        self.assertIn(-4.0, hours)
        self.assertIn(0.0, hours)


class WhiteNoiseTests(unittest.TestCase):
    def test_rho_zero_equals_closed_form_wls(self):
        times = [-1.0, 0.0, 2.0, 5.0]
        sig = [0.2, 0.1, 0.3, 0.15]
        y = [1.0, 1.2, 0.7, 1.5]
        f = fit(times, y, sig, [], 0.0, 1.0, 0.0, 1e-12)
        w = [1.0 / s ** 2 for s in sig]
        sw = sum(w)
        tm = sum(wi * ti for wi, ti in zip(w, times)) / sw
        ym = sum(wi * yi for wi, yi in zip(w, y)) / sw
        b1 = sum(wi * (ti - tm) * (yi - ym) for wi, ti, yi in zip(w, times, y)) \
            / sum(wi * (ti - tm) ** 2 for wi, ti in zip(w, times))
        b0 = ym - b1 * tm
        self.assertAlmostEqual(f['beta'][0], b0, places=12)
        self.assertAlmostEqual(f['beta'][1], b1, places=12)
        swt = sum(wi * ti for wi, ti in zip(w, times))
        swtt = sum(wi * ti * ti for wi, ti in zip(w, times))
        det = sw * swtt - swt * swt
        self.assertAlmostEqual(f['cov'][0][0], swtt / det, places=12)
        self.assertAlmostEqual(f['cov'][0][1], -swt / det, places=12)
        self.assertAlmostEqual(f['cov'][1][1], sw / det, places=12)

    def test_rho_zero_conditional_is_independent(self):
        tt = [0.0, 3.0]
        st = [0.2, 0.1]
        tv = [1.0, 2.0, 5.0]
        sv = [0.15, 0.25, 0.1]
        m, S = conditional_kernel([0.4, -0.3], tt, st, tv, sv, 0.0, 1.0)
        self.assertEqual(m, [0.0, 0.0, 0.0])
        for i in range(3):
            for j in range(3):
                self.assertEqual(S[i][j], sv[i] ** 2 if i == j else 0.0)
        rv = [0.1, -0.2, 0.5]
        nll = gaussian_nll(rv, S)
        direct = sum(0.5 * (math.log(2 * math.pi * sv[i] ** 2) + rv[i] ** 2 / sv[i] ** 2)
                     for i in range(3))
        self.assertAlmostEqual(nll, direct, places=12)


class MissingDataTests(unittest.TestCase):
    def test_gap_not_interpolated(self):
        times = [-20.0, -16.0, -12.0, -8.0, -4.0, 0.0, 18.0, 22.0, 26.0, 40.0, 44.0]
        v = dataset(times, wins=((-24, 0), (1, 30), (31, 48)))
        rep = analyze(v)
        hours = [r['hour'] for r in rep['development']['series']]
        hours += [r['hour'] for r in rep['external']['series']]
        self.assertEqual(sorted(hours), sorted(times))
        self.assertFalse(any(0.0 < h < 18.0 for h in hours))
        self.assertEqual(rep['identifiability']['status'], 'ok')


class IdentifiabilityTests(unittest.TestCase):
    def test_near_alias_flagged(self):
        rep = analyze(load('data/near-alias-frequency.json')[0])
        self.assertEqual(rep['identifiability']['status'], 'deficient')
        self.assertLess(rep['effective_rank']['full'], rep['effective_rank']['columns'])
        self.assertTrue(rep['effective_rank']['redundant_columns'])
        self.assertTrue(rep['identifiability']['near_collinear_pairs'])

    def test_insufficient_data_flagged(self):
        v = dataset([-1.0, 0.0, 1.0, 2.0], omegas=(0.5, 1.0),
                    candidates=[{'id': 'r', 'rho': 0.0, 'ridge': 0.5}],
                    wins=((-2, 0), (1, 1), (2, 2)))
        rep = analyze(v)
        self.assertEqual(rep['identifiability']['status'], 'deficient')
        self.assertLess(rep['effective_rank']['full'], rep['effective_rank']['columns'])
        self.assertTrue(any('记录数' in m for m in rep['identifiability']['messages']))

    def test_unpenalized_rank_deficiency_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            fit([2.0, 2.0], [1.0, 1.1], [0.2, 0.2], [0.5], 0.0, 1.0, 0.5, 1e-12)
        self.assertIn('未惩罚', str(ctx.exception))

    def test_all_candidates_fail_raises(self):
        v = dataset([0.0, 10.0, 20.0], wins=((-1, 0), (9, 11), (19, 21)))
        with self.assertRaises(ValueError) as ctx:
            analyze(v)
        self.assertIn('所有候选拟合失败', str(ctx.exception))


class PhaseTests(unittest.TestCase):
    def test_zero_amplitude_phase_unidentifiable(self):
        beta = [2.0, 0.1, 0.0, 0.0, 0.5, -0.5]
        cov = [[0.01 if i == j else 0.0 for j in range(6)] for i in range(6)]
        h = summarize_harmonics(beta, cov, [0.5, 1.0], 1e-8)
        self.assertIsNone(h[0]['phase_rad'])
        self.assertFalse(h[0]['phase_identifiable'])
        self.assertAlmostEqual(h[1]['amplitude'], math.hypot(0.5, -0.5))
        self.assertAlmostEqual(h[1]['phase_rad'], math.atan2(-0.5, 0.5))
        self.assertTrue(h[1]['phase_identifiable'])


class SelectionTests(unittest.TestCase):
    def test_tie_broken_by_candidate_id(self):
        v = dataset([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 30.0, 60.0],
                    candidates=[{'id': 'z', 'rho': 0.2, 'ridge': 0.1},
                                {'id': 'a', 'rho': 0.2, 'ridge': 0.1}],
                    wins=((-10, 4), (5, 40), (41, 80)))
        rep = analyze(v)
        scores = {c['id']: c['score_nll'] for c in rep['candidates']}
        self.assertEqual(scores['a'], scores['z'])
        self.assertEqual(rep['selected']['id'], 'a')

    def test_row_order_irrelevant(self):
        v, _ = load('data/normal.json')
        v2 = copy.deepcopy(v)
        v2['measurements'] = list(reversed(v2['measurements']))
        self.assertEqual(analyze(v)['development']['beta'],
                         analyze(v2)['development']['beta'])


class EndToEndTests(unittest.TestCase):
    def test_normal_json(self):
        v, _ = load('data/normal.json')
        rep = analyze(v)
        json.dumps(rep, allow_nan=False)
        self.assertIn(rep['selected']['id'], ('white', 'correlated', 'regularized'))
        dev = rep['development']
        self.assertEqual(dev['n'], 16)
        self.assertEqual(len(rep['external']['series']), 4)
        for r in dev['series']:
            self.assertAlmostEqual(r['level'], r['trend'] + r['tidal'] + r['residual'], places=9)
        for r in rep['external']['series']:
            self.assertAlmostEqual(r['residual'], r['level'] - r['prediction'], places=9)
        self.assertEqual(rep['identifiability']['status'], 'ok')
        self.assertEqual(rep['effective_rank']['full'], 6)
        for h in dev['harmonics']:
            self.assertTrue(h['phase_identifiable'])

    def test_prediction_difference_variance(self):
        rep = analyze(load('data/normal.json')[0])
        dev = rep['development']
        P = dev['prediction_covariance']
        cov = dev['beta_covariance']
        om = rep['frequencies']
        hours = [r['hour'] for r in dev['series']]
        i, j = 0, 7
        xi = design([hours[i]], om)[0]
        xj = design([hours[j]], om)[0]
        diff = [a - b for a, b in zip(xi, xj)]
        var = sum(diff[a] * cov[a][b] * diff[b] for a in range(6) for b in range(6))
        self.assertAlmostEqual(P[i][i] + P[j][j] - 2 * P[i][j], var, places=10)
        self.assertGreater(var, 0.0)
        self.assertNotAlmostEqual(P[i][j], 0.0)

    def test_same_time_repeat_merged(self):
        rep = analyze(load('data/same-time-repeat.json')[0])
        self.assertEqual(rep['development']['n'], 16)

    def test_conflict_and_invalid_rejected(self):
        for name in ('invalid-same-time-conflict', 'invalid-domain'):
            with self.subTest(file=name):
                with self.assertRaises(ValueError):
                    analyze(load('data/%s.json' % name)[0])


class PrepareTests(unittest.TestCase):
    def test_window_overlap_rejected(self):
        v = dataset([0.0, 1.0, 60.0], wins=((-5, 5), (1, 50), (51, 100)))
        with self.assertRaises(ValueError):
            prepare(v)

    def test_time_outside_windows_rejected(self):
        v = dataset([0.0, 1.0, 200.0])
        with self.assertRaises(ValueError):
            prepare(v)

    def test_bad_epoch_rejected(self):
        v = dataset([0.0, 1.0, 60.0])
        v['epoch'] = '2020-01-01 00:00'
        with self.assertRaises(ValueError):
            prepare(v)


if __name__ == '__main__':
    unittest.main()
