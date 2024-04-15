import taichi as ti

ti.init(arch=ti.gpu)

# Parameters
num_particles = 10000
dt = 1e-3
substeps = 10

# Particle properties
x = ti.Vector.field(3, dtype=float, shape=num_particles)  # 3D position
v = ti.Vector.field(3, dtype=float, shape=num_particles)  # 3D velocity

# Gravity
g = ti.Vector([0, -9.8, 0])  # Gravity in 3D


@ti.kernel
def initialize():
    for i in range(num_particles):
        x[i] = [0.5 + ti.random() * 0.2, 0.5 + ti.random() * 0.2,
                0.5 + ti.random() * 0.2]  # 3D position
        v[i] = [-1 + ti.random() * 2, -1 + ti.random() * 2, -
                1 + ti.random() * 2]  # 3D velocity


@ti.kernel
def substep():
    for i in range(num_particles):
        v[i] += dt * g
        x[i] += dt * v[i]


window = ti.ui.Window("Taichi Shotcrete", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()
camera = ti.ui.Camera()

current_t = 0.0
initialize()

while window.running:
    if current_t > 1.5:
        # Reset
        initialize()
        current_t = 0

    for i in range(substeps):
        substep()
        current_t += dt

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))

    particles = x.to_numpy()
    for particle in particles:
        scene.particles(particle, radius=3, color=0x068587)

    canvas.scene(scene)
    window.show()
