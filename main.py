import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib

from src.sampling import (
    generate_bootstrap_samples,
    generate_subsamples
)

def main():
    print("=" * 50)
    print("UCZENIE ZESPOLOWE - ANALIZA BAGGINGU")
    print("=" * 50)

    print("\nTEST PRÓBKOWANIA")

    n = 1000

    bootstrap_samples = generate_bootstrap_samples(
        n=n,
        ratio=1.0,
        n_estimators=100,
        random_state=42
    )

    subsamples = generate_subsamples(
        n=n,
        ratio=0.63,
        n_estimators=100,
        random_state=42
    )

    print("Liczba próbek bootstrap:", len(bootstrap_samples))
    print("Rozmiar pierwszej próbki:", len(bootstrap_samples[0]))
    print("Unikalne obserwacje w bootstrap:",
          len(np.unique(bootstrap_samples[0])))

    print("\nLiczba podpróbek:", len(subsamples))
    print("Rozmiar pierwszej podpróbki:", len(subsamples[0]))
    print("Unikalne obserwacje w podpróbce:",
          len(np.unique(subsamples[0])))

if __name__ == "__main__":
    main()