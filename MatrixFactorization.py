import os
import numpy as np
import argparse


class Factorizer:
    def __init__(
        self,
        src_path,
        dst_path,
        k,
        alpha=0.01,
        beta=0.002,
        epsilon=1e-6,
        symmetric=False,
        n_steps=1000,
        log_every=10,
        save_every=100,
    ):
        self.R = np.load(args.src_path)
        self.n, self.m = self.R.shape
        self.k = k
        self.P = np.random.rand(self.n, self.k)
        self.Q = np.random.rand(self.k, self.m)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.symmetric = symmetric
        self.dst_path = dst_path
        self.n_steps = n_steps
        self.log_every = log_every
        self.save_every = save_every

    def __call__(self):
        print(f"running algorithm for {self.n_steps} steps")
        for step in range(self.n_steps):
            self.run_step()
            err, reg = self.get_error()
            if err + reg < self.epsilon:
                break
            if step and step % self.save_every == 0:
                s = os.path.splitext(self.dst_path)
                dst_path = s[0] + f"_{step}" + s[1]
                print(f"saving into {dst_path}")
                np.save(dst_path, np.dot(self.P, self.Q))
            if step % self.log_every == 0:
                print(f"step {step}, error: {err}, regularized: {err+reg}")

        nR = np.dot(self.P, self.Q)
        result = np.zeros((self.n, self.m))
        for i in range(self.n):
            for j in range(i + 1 if self.symmetric else self.m):
                if self.R[i][j] == -1:
                    result[i][j] = nR[i][j]
                else:
                    result[i][j] = self.R[i][j]
        print(f"saving into {self.dst_path}")
        np.savetxt(self.dst_path, result)

    def run_step(self):
        for i in range(self.n):
            for j in range(i + 1 if self.symmetric else self.m):
                if self.R[i][j] >= 0:
                    eij = self.R[i][j] - np.dot(self.P[i, :], self.Q[:, j])
                    self.P[i] += (
                        self.alpha * 2 * eij * self.Q[:, j] - self.beta * self.P[i]
                    )
                    self.Q[:, j] += (
                        self.alpha * 2 * eij * self.P[i] - self.beta * self.Q[:, j]
                    )

    def get_error(self):
        err = 0
        reg = 0
        for i in range(self.n):
            for j in range(i + 1 if self.symmetric else self.m):
                if self.R[i][j] > 0:
                    err += pow(self.R[i][j] - np.dot(self.P[i, :], self.Q[:, j]), 2)
                    reg += np.sum(
                        self.beta
                        / 2
                        * (np.power(self.P[i], 2) + np.power(self.Q[:, j], 2))
                    )
        return err, reg


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Running matrix focatorization.")
    parser.add_argument("--src-path", help="input file", required=True, type=str)
    parser.add_argument("--dst-path", help="output file", required=False, type=str)
    parser.add_argument(
        "--n-dim", help="number of dimensions for latent space", required=True, type=int
    )
    parser.add_argument("--alpha", required=False, type=float, default=0.002)
    parser.add_argument("--beta", required=False, type=float, default=0.002)
    parser.add_argument(
        "--n-steps", help="number of steps", required=False, type=int, default=1000
    )
    parser.add_argument(
        "--log-every", help="how often to log", required=False, type=int, default=10
    )
    parser.add_argument(
        "--save-every", help="how often to save", required=False, type=int, default=100
    )

    parser.add_argument(
        "--epsilon", help="target error", required=False, type=float, default=10
    )
    parser.add_argument(
        "--symmetric", help="whether input is symmetric", action="store_true"
    )

    args = parser.parse_args()

    dst_path = args.dst_path
    if dst_path is None:
        name, ext = os.path.splitext(args.src_path)
        dst_path = (
            name + f"_d{args.n_dim}_a{str(args.alpha)[2:]}_b{str(args.beta)[2:]}" + ext
        )

    factorizer = Factorizer(
        src_path=args.src_path,
        dst_path=dst_path,
        k=args.n_dim,
        alpha=args.alpha,
        beta=args.beta,
        epsilon=args.epsilon,
        symmetric=args.symmetric,
        n_steps=args.n_steps,
        log_every=args.log_every,
        save_every=args.save_every,
    )

    factorizer()
