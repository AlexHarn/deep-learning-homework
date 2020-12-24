import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 10000)


def f(x):
    return -x*x + 1.


plt.plot(x, f(x), label=r'$f(x)$')
plt.step([-1, 1, 1.0001], [0, np.mean(f(x)), 0], label=r'$f_1(x)$')
plt.grid()
plt.legend()
plt.savefig('0.png')
plt.clf()


def gen_steps_and_means(N):
    d = 2/N
    steps = [-1 + d*i for i in range(N)]
    means = []
    for step in steps:
        x = np.linspace(step, step+d, 100)
        means.append(np.mean(f(x)))
    steps = steps + [1, 1.0001]
    means = [0] + means + [0]
    return steps, means


steps, means = gen_steps_and_means(3)

for i in range(1, 4):
    plt.plot(x, f(x), label=r'$f(x)$')
    plt.step(steps[:i+1]+[steps[-1]], means[:i+1]+[means[-1]], label=r'$f_{}(x)$'.format(i))
    plt.grid()
    plt.legend()
    plt.savefig('{}.png'.format(i))
    plt.clf()

for N in [10, 20, 50, 100, 1000]:
    steps, means = gen_steps_and_means(N)
    plt.plot(x, f(x), label=r'$f(x)$')
    plt.step(steps, means, label=r'$f_{'+ str(N) + r'}(x)$')
    plt.grid()
    plt.legend()
    plt.savefig('{}.png'.format(N))
    plt.clf()
