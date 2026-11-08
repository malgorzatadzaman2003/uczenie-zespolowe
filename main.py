import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib


def main():
    print("=" * 50)
    print("UCZENIE ZESPOLOWE - ANALIZA BAGGINGU")
    print("=" * 50)

    print(f"NumPy: {np.version}")
    print(f"Pandas: {pd.version}")
    print(f"Scikit-learn: {sklearn.version}")
    print(f"SciPy: {scipy.version}")
    print(f"Matplotlib: {matplotlib.version}")

    print("\nŚrodowisko projektu działa poprawnie!")

if __name__ == "__main__":
    main()