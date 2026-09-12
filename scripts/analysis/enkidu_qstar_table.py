def q_star(eta, tol):
    return eta * (1 - eta - tol) / ((eta + tol) * (1 - eta))

def main():
    eta = 0.05
    print("ENKIDU interface reference table, eta=%.2f" % eta)
    print("%-12s %-18s %-10s" % ("tolerance", "target_fpr_ceiling", "q_star"))
    for tol in [0.05, 0.10, 0.20, 0.45]:
        ceiling = eta + tol
        q = q_star(eta, tol)
        print("%-12.2f %-18.2f %-10.4f" % (tol, ceiling, q))

if __name__ == "__main__":
    main()
