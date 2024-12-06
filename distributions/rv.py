import numpy as np
from scipy.stats import rv_continuous


class a_gen(rv_continuous):
    """
    Distribuição A.

    A distribuição A é definida com os seguintes métodos:
    - PDF: Função de densidade de probabilidade.
    - CDF: Função de distribuição acumulada.
    - PPF: Função inversa da CDF (quantil).
    - Log-PDF: Logaritmo da função de densidade de probabilidade.
    - Log-CDF: Logaritmo da função de distribuição acumulada.
    """

    def _pdf(self, x, beta):
        """
        Função de densidade de probabilidade (PDF) da distribuição A.

        A fórmula para o PDF é dada por:
            f_X(x; beta) = exp( (1/beta) * (1 - exp(beta/x)) + beta/x ) / (x^2)

        Parâmetros:
            x : array-like
                Os valores de x onde o PDF será calculado.
            beta : float
                O parâmetro beta da distribuição A.

        Retorna:
            PDF : array-like
                O valor da função de densidade de probabilidade para cada x.
        """
        return np.exp((1 / beta) * (1 - np.exp(beta / x)) + beta / x) / (x ** 2)

    def _cdf(self, x, beta):
        """
        Função de distribuição acumulada (CDF) da distribuição A.

        A fórmula para a CDF é dada por:
            F_X(x; beta) = exp( (1/beta) * (1 - exp(beta/x)) )

        Parâmetros:
            x : array-like
                Os valores de x onde a CDF será calculada.
            beta : float
                O parâmetro beta da distribuição A.

        Retorna:
            CDF : array-like
                O valor da função de distribuição acumulada para cada x.
        """
        return np.exp((1 / beta) * (1 - np.exp(beta / x)))

    def _ppf(self, q, beta):
        """
        Função inversa da CDF (quantil) da distribuição A.

        A fórmula para o PPF é dada por:
            PPF_X(q; beta) = beta / log(1 - beta * log(q))

        Parâmetros:
            q : array-like
                Os quantis (valores de probabilidade) onde o PPF será calculado.
            beta : float
                O parâmetro beta da distribuição A.

        Retorna:
            Quantil : array-like
                O valor do quantil para cada q.
        """
        return beta / np.log(1 - beta * np.log(q))

    def _logpdf(self, x, beta):
        """
        Logaritmo da função de densidade de probabilidade (log-pdf) da distribuição A.

        A fórmula para o log-pdf é dada por:
            log f_X(x; beta) = beta/x - 2 * log(x) + (1 - exp(beta/x)) / beta

        Parâmetros:
            x : array-like
                Os valores de x onde o log-pdf será calculado.
            beta : float
                O parâmetro beta da distribuição A.

        Retorna:
            Log-PDF : array-like
                O valor do logaritmo da função de densidade de probabilidade para cada x.
        """
        return beta / x - 2 * np.log(x) + (1 - np.exp(beta / x)) / beta

    def _logcdf(self, x, beta):
        """
        Logaritmo da função de distribuição acumulada (log-cdf) da distribuição A.

        A fórmula para o log-cdf é dada por:
            log F_X(x; beta) = (1 / beta) * (1 - exp(beta/x))

        Parâmetros:
            x : array-like
                Os valores de x onde o log-cdf será calculado.
            beta : float
                O parâmetro beta da distribuição A.

        Retorna:
            Log-CDF : array-like
                O valor do logaritmo da função de distribuição acumulada para cada x.
        """
        return (1 / beta) * (1 - np.exp(beta / x))


# Instanciando a distribuição A com parâmetro beta e suporte nos reais positivos
a = a_gen(a=0.0, name='A')
