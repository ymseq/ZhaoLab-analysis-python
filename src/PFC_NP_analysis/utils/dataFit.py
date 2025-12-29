

import numpy as np
from scipy import stats


# ========================
# 1. 定义链接函数和分布族
# ========================

class Family:
    """指数族基类"""
    def link(self, mu):
        raise NotImplementedError

    def inverse_link(self, eta):
        raise NotImplementedError

    def deta_dmu(self, mu):
        """d eta / d mu"""
        raise NotImplementedError

    def variance(self, mu):
        """Var(Y | mu)"""
        raise NotImplementedError

    def deviance(self, y, mu):
        """Deviance"""
        raise NotImplementedError


class Gaussian(Family):
    """高斯分布 + identity link (普通线性回归)"""
    def link(self, mu):
        return mu

    def inverse_link(self, eta):
        return eta

    def deta_dmu(self, mu):
        return np.ones_like(mu)

    def variance(self, mu):
        # 常数方差, 这里取 1（真实 σ^2 会作为尺度参数）
        return np.ones_like(mu)

    def deviance(self, y, mu):
        # 对高斯而言, deviance 与 RSS 成正比
        return np.sum((y - mu) ** 2)


class Binomial(Family):
    """二项分布 + logit link (Logistic 回归, y ∈ {0,1})"""
    def link(self, mu):
        eps = 1e-9
        mu = np.clip(mu, eps, 1 - eps)
        return np.log(mu / (1 - mu))

    def inverse_link(self, eta):
        # logistic 函数
        return 1.0 / (1.0 + np.exp(-eta))

    def deta_dmu(self, mu):
        eps = 1e-9
        mu = np.clip(mu, eps, 1 - eps)
        return 1.0 / (mu * (1 - mu))

    def variance(self, mu):
        return mu * (1 - mu)

    def deviance(self, y, mu):
        eps = 1e-9
        y = np.clip(y, eps, 1 - eps)
        mu = np.clip(mu, eps, 1 - eps)
        return 2 * np.sum(
            y * np.log(y / mu) +
            (1 - y) * np.log((1 - y) / (1 - mu))
        )


class Poisson(Family):
    """泊松分布 + log link"""
    def link(self, mu):
        eps = 1e-9
        mu = np.clip(mu, eps, None)
        return np.log(mu)

    def inverse_link(self, eta):
        return np.exp(eta)

    def deta_dmu(self, mu):
        eps = 1e-9
        mu = np.clip(mu, eps, None)
        return 1.0 / mu

    def variance(self, mu):
        # Var(Y|mu) = mu
        return mu

    def deviance(self, y, mu):
        eps = 1e-9
        y = np.clip(y, eps, None)
        mu = np.clip(mu, eps, None)
        return 2 * np.sum(y * np.log(y / mu) - (y - mu))


# ========================
# 2. GLM 主类 & 结果对象
# ========================

class GLMResult:
    def __init__(self, coef_, family, mu_, eta_, deviance_, df_model, df_resid, nobs, converged, name=None):
        self.coef_ = coef_
        self.family = family
        self.mu_ = mu_
        self.eta_ = eta_
        self.deviance = deviance_
        self.df_model = df_model
        self.df_resid = df_resid
        self.nobs = nobs
        self.converged = converged
        self.name = name if name is not None else "GLM"

    def predict(self, X, add_intercept=True):
        if add_intercept:
            X = add_intercept_column(X)
        eta = X @ self.coef_
        return self.family.inverse_link(eta)

    def summary(self):
        print(f"Model: {self.name}")
        print(f"Family: {self.family.__class__.__name__}")
        print(f"nobs: {self.nobs}")
        print(f"df_model: {self.df_model}")
        print(f"df_resid: {self.df_resid}")
        print(f"Deviance: {self.deviance:.4f}")
        print(f"Converged: {self.converged}")
        print("Coefficients:")
        for i, c in enumerate(self.coef_):
            print(f"  beta[{i}] = {c:.6f}")


def add_intercept_column(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    intercept = np.ones((n, 1))
    return np.hstack([intercept, X])


class GLM:
    def __init__(self, family, fit_intercept=True, max_iter=100, tol=1e-6, name=None):
        """
        family: Family 子类实例, 如 Gaussian(), Binomial(), Poisson()
        fit_intercept: 是否自动加截距
        """
        self.family = family
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.name = name if name is not None else "GLM"

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples = X.shape[0]

        if self.fit_intercept:
            X = add_intercept_column(X)

        n_features = X.shape[1]

        # 初始值：beta = 0
        beta = np.zeros(n_features)
        converged = False

        for it in range(self.max_iter):
            eta = X @ beta                       # 线性预测
            mu = self.family.inverse_link(eta)   # 期望 mu

            # 计算导数和方差
            deta_dmu = self.family.deta_dmu(mu)
            dmu_deta = 1.0 / deta_dmu
            var_mu = self.family.variance(mu)

            # IRLS 权重和工作响应
            w = (dmu_deta ** 2) / var_mu        # weights
            z = eta + (y - mu) * deta_dmu       # working response

            # 防止权重为 0
            w = np.clip(w, 1e-9, np.inf)

            # 加权最小二乘： (X^T W X) beta = X^T W z
            WX = X * w[:, None]
            XtWX = X.T @ WX
            XtWz = X.T @ (w * z)

            try:
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                # 奇异矩阵时使用最小二乘
                beta_new, *_ = np.linalg.lstsq(XtWX, XtWz, rcond=None)

            # 收敛判定
            if np.max(np.abs(beta_new - beta)) < self.tol:
                beta = beta_new
                converged = True
                break

            beta = beta_new

        # 拟合后的量
        eta = X @ beta
        mu = self.family.inverse_link(eta)
        deviance = self.family.deviance(y, mu)

        df_model = n_features - 1 if self.fit_intercept else n_features
        df_resid = n_samples - n_features

        result = GLMResult(
            coef_=beta,
            family=self.family,
            mu_=mu,
            eta_=eta,
            deviance_=deviance,
            df_model=df_model,
            df_resid=df_resid,
            nobs=n_samples,
            converged=converged,
            name=self.name
        )
        return result


# ========================
# 3. GLM 方差分析（Deviance ANOVA）
# ========================

def anova_glm(*models):
    """
    对一组嵌套的 GLM 模型做 deviance 方差分析（LRT）
    参数：传入若干 GLMResult 对象，默认按 df_model 排序
    返回：一个列表，每行包含：
        - model_name
        - df_model
        - deviance
        - df_diff
        - deviance_diff
        - p_value
    """
    if len(models) < 2:
        raise ValueError("anova_glm 至少需要两个模型（嵌套模型）")

    # 按自变量个数排序（小 -> 大），对应：简单模型 -> 复杂模型
    models_sorted = sorted(models, key=lambda m: m.df_model)
    rows = []

    prev = models_sorted[0]
    rows.append({
        "model": prev.name,
        "df_model": prev.df_model,
        "deviance": prev.deviance,
        "df_diff": np.nan,
        "deviance_diff": np.nan,
        "p_value": np.nan
    })

    for m in models_sorted[1:]:
        df_diff = m.df_model - prev.df_model
        dev_diff = prev.deviance - m.deviance
        # 似然比检验：统计量 ~ Chi^2(df_diff)
        p_value = stats.chi2.sf(dev_diff, df_diff)
        rows.append({
            "model": m.name,
            "df_model": m.df_model,
            "deviance": m.deviance,
            "df_diff": df_diff,
            "deviance_diff": dev_diff,
            "p_value": p_value
        })
        prev = m

    # 打印简单表格
    header = f"{'Model':15s} {'df_model':8s} {'Deviance':10s} {'df_diff':8s} {'Dev.diff':10s} {'Pr(>Chi)':10s}"
    print(header)
    print("-" * len(header))

    for r in rows:
        # 先把可能是 NaN 的数格式化成字符串
        if np.isnan(r["deviance_diff"]):
            dev_diff_str = ""
        else:
            dev_diff_str = f"{r['deviance_diff']:.4f}"

        if np.isnan(r["p_value"]):
            pval_str = ""
        else:
            pval_str = f"{r['p_value']:.4g}"

        df_diff_str = "" if np.isnan(r["df_diff"]) else str(int(r["df_diff"]))

        print(
            f"{r['model']:15s} "
            f"{str(r['df_model']):8s} "
            f"{r['deviance']:.4f}    "
            f"{df_diff_str:8s} "
            f"{dev_diff_str:10s} "
            f"{pval_str:10s}"
        )
    return rows












