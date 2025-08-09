import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# images klasörü yoksa oluştur
os.makedirs("images", exist_ok=True)

# ------------------------
# 1. BLOCH SPHERE GÖRSELİ
# ------------------------
def plot_bloch_sphere(state_vector, filename):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    # Küre çizimi
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='c', alpha=0.1, edgecolor='w')

    # Eksenler
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1) # X
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1) # Y
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1) # Z

    # State vector koordinatları
    theta = 2 * np.arccos(np.abs(state_vector[0]))
    phi = np.angle(state_vector[1]) - np.angle(state_vector[0])

    xs = np.sin(theta) * np.cos(phi)
    ys = np.sin(theta) * np.sin(phi)
    zs = np.cos(theta)

    ax.quiver(0, 0, 0, xs, ys, zs, color='k', arrow_length_ratio=0.1, linewidth=2)

    ax.set_box_aspect([1,1,1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("Bloch Sphere", fontsize=16)

    plt.savefig(filename, dpi=300)
    plt.close()

# Örnek state: |ψ> = (|0> + i|1>)/√2
state_example = np.array([1/np.sqrt(2), 1j/np.sqrt(2)])
plot_bloch_sphere(state_example, "images/bloch_sphere_example.png")

# ------------------------
# 2. TELEPORTATION CIRCUIT
# ------------------------
import matplotlib.pyplot as plt

def draw_teleportation_circuit(filename):
    fig, ax = plt.subplots(figsize=(8,4))

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')

    # Qubit çizgileri
    for i, label in enumerate(["q0: |ψ>", "q1: |0>", "q2: |0>"]):
        ax.hlines(y=i, xmin=0, xmax=10, color='black')
        ax.text(-0.5, i, label, fontsize=12, va='center')

    # Kapılar (basit sembolik gösterim)
    # Hadamard
    ax.add_patch(plt.Rectangle((1, 1-0.3), 0.6, 0.6, fill=True, color='lightblue'))
    ax.text(1.3, 1, "H", ha='center', va='center', fontsize=14)

    # CNOT q1 -> q2
    ax.plot(2, 1, 'ko', markersize=8)
    ax.plot(2, 2, 'ko', markersize=8)
    ax.vlines(2, 1, 2, color='black')
    ax.plot(2, 2, 'o', color='white', markersize=12, markeredgecolor='black')
    ax.plot(2.1, 2, '+', color='black', markersize=12)

    # CNOT q0 -> q1
    ax.plot(4, 0, 'ko', markersize=8)
    ax.plot(4, 1, 'ko', markersize=8)
    ax.vlines(4, 0, 1, color='black')
    ax.plot(4, 1, 'o', color='white', markersize=12, markeredgecolor='black')
    ax.plot(4.1, 1, '+', color='black', markersize=12)

    ax.set_title("Quantum Teleportation Circuit", fontsize=14)
    plt.savefig(filename, dpi=300)
    plt.close()

draw_teleportation_circuit("images/teleportation_circuit_example.png")

print("Görseller images/ klasörüne kaydedildi.")
