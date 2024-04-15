import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

N = 15000
a_numpy = np.random.randint(0, 100, N, dtype=np.int32)
b_numpy = np.random.randint(0, 100, N, dtype=np.int32)

f = ti.field(dtype=ti.i32, shape=(N + 1, N + 1))


@ti.kernel
def compute_lcs(a: ti.types.ndarray(), b: ti.types.ndarray()) -> ti.i32:
    len_a, len_b = a.shape[0], b.shape[0]

    ti.loop_config(serialize=True)  # Disable auto-parallelism in Taichi
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            f[i, j] = ti.max(f[i - 1, j - 1] + (a[i - 1] == b[j - 1]),
                             ti.max(f[i - 1, j], f[i, j - 1]))

    return f[len_a, len_b]


print(compute_lcs(a_numpy, b_numpy))
