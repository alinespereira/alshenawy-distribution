import pymc as pm
import jax.numpy as jnp
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.shape_utils import rv_size_is_none

from distributions.ops import a_rv  # Certifique-se de que `a_rv` esteja implementado corretamente

# Subclassificando PositiveContinuous para definir uma distribuição customizada
class A(PositiveContinuous):
    # Este será usado pela metaclasse `DistributionMeta` para despachar
    # os métodos logp e logcdf para o RandomVariable (Op) customizado definido abaixo.
    rv_op = a_rv

    @classmethod
    def dist(cls, beta, **kwargs):
        """
        Retorna uma instância da distribuição usando os parâmetros fornecidos.

        Parâmetros:
            beta: float
                O parâmetro beta da distribuição A.

        Retorna:
            Instância da distribuição com o parâmetro beta.
        """
        beta = jnp.array(beta)  # Converte beta para um array do JAX para compatibilidade
        return super().dist([beta], **kwargs)

    def support_point(rv, size, beta):
        """
        Retorna uma expressão simbólica para o ponto representativo (ex: média ou moda)
        de onde começar a amostragem.

        Parâmetros:
            rv: RandomVariable
                O RandomVariable associado à distribuição.
            size: Tupla
                O tamanho da amostra.
            beta: float
                O parâmetro beta da distribuição A.

        Retorna:
            support_point: array
                Ponto representativo da amostra.
        """
        support_point = jnp.broadcast_to(beta, size)
        return support_point

    def logp(value, beta):
        """
        Calcula o logaritmo da função de densidade de probabilidade (log-pdf) para a distribuição A.

        A fórmula para o log-pdf é:
            log f_X(x; beta) = beta/x - 2 * log(x) + (1 - exp(beta/x)) / beta

        Parâmetros:
            value: array-like
                O valor de x onde o log-pdf será calculado.
            beta: float
                O parâmetro beta da distribuição A.

        Retorna:
            logp_expression: array-like
                O valor do log-pdf calculado para cada valor de x e beta.
        """
        # Expressão do log-pdf baseada na fórmula da distribuição A
        logp_expression = (
            -2 * jnp.log(value)
            + (1 / beta) * (1 - jnp.exp(beta / value))
            + beta / value
        )

        # Usamos jnp.where para impor o domínio de suporte (value > 0)
        bounded_logp_expression = jnp.where(
            value > 0, logp_expression, -jnp.inf,
        )

        # Validação dos parâmetros (garante que beta seja positivo)
        return check_parameters(
            bounded_logp_expression,
            beta > 0,
            msg="beta deve ser maior que 0",  # Mensagem de erro caso a condição falhe
        )

    def logcdf(value, beta):
        """
        Logaritmo da função de distribuição acumulada (log-CDF) da distribuição A.

        A fórmula para o log-CDF é dada por:
            log F_X(x; beta) = (1 / beta) * (1 - exp(beta / x))

        Parâmetros:
            value : array-like
                Os valores de x onde o log-cdf será calculado.
            beta : float
                O parâmetro beta da distribuição A.

        Retorna:
            Log-CDF : array-like
                O valor do logaritmo da função de distribuição acumulada para cada x.
        """
        # Expressão para a log-CDF
        logcdf_expression = (1 / beta) * (1 - jnp.exp(beta / value))

        # Garantir que a expressão seja válida para x > 0
        bounded_logcdf_expression = jnp.where(
            value > 0, logcdf_expression, -jnp.inf
        )

        # Verificar se o parâmetro beta é válido (beta > 0)
        return check_parameters(
            bounded_logcdf_expression,
            beta > 0,
            msg="beta deve ser maior que 0",  # Mensagem de erro caso a condição falhe
        )
