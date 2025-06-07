import random
from sympy import symbols, mod_inverse
from itertools import combinations

# -------------------------------------------------------
# 1. パラメータ設定
# -------------------------------------------------------
x, y = symbols('x y')

n = 5          # 参加者数
t = 4          # 初期閾値
t_prime = 2    # 新閾値（t_prime < t とする）
p = 208351617316091241234326746312124448251235562226470491514186331217050270460481  # 大きな素数
y0 = 123456    # 新閾値更新時に公開する y0

# -------------------------------------------------------
# 2. 双変数多項式 F(x,y) の係数構造を構築
#    F(x,y) = ∑_{k=0..t'-1} a_k * x^k + ∑_{k=t'..t-1} b_k * (y - y0) * x^k
# -------------------------------------------------------

# 2-1. 秘密 s を乱数で生成
s = random.randrange(1, p)

# 2-2. k=0..t'-1 の a_k (定数項)
a = {}  # a[k] は x^k の定数係数 (0 ≤ k < t')
for k in range(t_prime):
    if k == 0:
        a[k] = s
    else:
        a[k] = random.randrange(0, p)

# 2-3. k=t'..t-1 の b_k (ランダム定数)
b = {}  # b[k] は x^k に対して (y - y0) をかける係数, (t' ≤ k < t)
for k in range(t_prime, t):
    b[k] = random.randrange(0, p)

# -------------------------------------------------------
# 3. 各参加者 i=1..n に対して S_i(y) = F(i, y) => shares[i] = (u_i, v_i)
# -------------------------------------------------------

shares = {}
for i in range(1, n+1):
    # (1) u_i の計算 (y=0 を代入したときの定数項)
    sum_a = sum(a[k] * pow(i, k, p) for k in range(t_prime)) % p
    sum_b_y0 = sum(b[k] * pow(i, k, p) for k in range(t_prime, t)) % p
    u_i = (sum_a - (y0 * sum_b_y0) % p) % p

    # (2) v_i の計算 (y にかかる係数)
    v_i = sum(b[k] * pow(i, k, p) for k in range(t_prime, t)) % p

    shares[i] = (u_i, v_i)

# -------------------------------------------------------
# 4. Shamir 補間用関数 (一次元)
# -------------------------------------------------------

def lagrange_univariate(x_list, y_list, x0, p):
    total = 0
    k = len(x_list)
    for i in range(k):
        xi, yi = x_list[i], y_list[i]
        num, den = 1, 1
        for j in range(k):
            if i == j:
                continue
            xj = x_list[j]
            num = (num * (x0 - xj)) % p
            den = (den * (xi - xj)) % p
        li = (num * mod_inverse(den, p)) % p
        total = (total + yi * li) % p
    return total

# -------------------------------------------------------
# 5. 復元テスト
#    - 初期閾値 t でのテスト: (t-1)=3 人で失敗→ t=4 人で成功
#    - 新しい閾値 t' でのテスト: (t'-1)=1 人で失敗→ t'=2 人で成功
# -------------------------------------------------------

# テスト結果格納用
results = {
    "initial_failure": [],  # 初期閾値で (t-1) 人集めた場合
    "initial_success": [],  # 初期閾値で t 人集めた場合
    "new_failure": [],      # 新しい閾値で (t'-1) 人集めた場合
    "new_success": []       # 新しい閾値で t' 人集めた場合
}

# (A) 初期閾値 t のテスト: (t-1)=3 人で集めた場合に失敗するか
for combo in combinations(range(1, n+1), t-1):
    xs = list(combo)
    ys = [shares[i][0] for i in combo]  # u_i (y=0 の古いシャドウ)
    recovered = lagrange_univariate(xs, ys, 0, p)
    results["initial_failure"].append((combo, recovered, recovered != s))

# (B) 初期閾値 t のテスト: t=4 人で集めた場合に成功するか
for combo in combinations(range(1, n+1), t):
    xs = list(combo)
    ys = [shares[i][0] for i in combo]
    recovered = lagrange_univariate(xs, ys, 0, p)
    results["initial_success"].append((combo, recovered, recovered == s))

# (C) 新しい閾値 t' のテスト: (t'-1)=1 人で集めた場合に失敗するか
for combo in combinations(range(1, n+1), t_prime-1):
    xs = list(combo)
    ys = [(shares[i][0] + shares[i][1] * y0) % p for i in combo]  # S_i(y0)
    recovered = lagrange_univariate(xs, ys, 0, p)
    results["new_failure"].append((combo, recovered, recovered != s))

# (D) 新しい閾値 t' のテスト: t'=2 人で集めた場合に成功するか
for combo in combinations(range(1, n+1), t_prime):
    xs = list(combo)
    ys = [(shares[i][0] + shares[i][1] * y0) % p for i in combo]
    recovered = lagrange_univariate(xs, ys, 0, p)
    results["new_success"].append((combo, recovered, recovered == s))

# -------------------------------------------------------
# 6. 結果表示
# -------------------------------------------------------
print(f"秘密 s = {s}\n")

print("=== 初期閾値 t のテスト ===")
print(f"--- (t-1)={t-1} 人で集めた場合 (期待: 失敗) ---")
for combo, rec, is_failure in results["initial_failure"]:
    print(f"組み合わせ {combo} -> 復元値: {rec} (失敗している: {is_failure})")

print(f"\n--- t={t} 人で集めた場合 (期待: 成功) ---")
for combo, rec, is_success in results["initial_success"]:
    print(f"組み合わせ {combo} -> 復元値: {rec} (成功している: {is_success})")

print("\n=== 新しい閾値 t' のテスト ===")
print(f"--- (t'-1)={t_prime-1} 人で集めた場合 (期待: 失敗) ---")
for combo, rec, is_failure in results["new_failure"]:
    print(f"組み合わせ {combo} -> 復元値: {rec} (失敗している: {is_failure})")

print(f"\n--- t'={t_prime} 人で集めた場合 (期待: 成功) ---")
for combo, rec, is_success in results["new_success"]:
    print(f"組み合わせ {combo} -> 復元値: {rec} (成功している: {is_success})")
