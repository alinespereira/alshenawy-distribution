from typing import Tuple

import jax
import jax.numpy as jnp
from pytensor.tensor.random.op import RandomVariable

from distributions.rv import a


class ARV(RandomVariable):
    """
    RandomVariable personalizado para a distribuição A.
    """
    name: str = "a"

    # A assinatura NumPy/PyTensor para este RV
    signature = "()->()"

    # O tipo de dado (dtype) para este RV, usando JAX
    dtype: str = "float32"

    # Nome e representação LaTeX para o RV
    _print_name: Tuple[str, str] = ("A", "\\operatorname{A}")

    @classmethod
    def rng_fn(
        cls,
        rng_key,  # Usando o JAX RNG Key
        beta: jnp.ndarray,
        size: Tuple[int, ...],
    ) -> jnp.ndarray:
        """
        Função para gerar amostras da distribuição A usando JAX.

        Parâmetros:
            rng_key: jax.random.KeyArray
                A chave do gerador de números aleatórios do JAX.
            beta: jnp.ndarray
                O parâmetro beta para a distribuição A.
            size: Tupla
                O tamanho da amostra a ser gerada.

        Retorna:
            jnp.ndarray
                Amostras geradas de acordo com a distribuição A.
        """
        return a.rvs(beta, random_state=rng_key, size=size)

# Criando o RandomVariable 'Op' para a distribuição A
a_rv = ARV()
