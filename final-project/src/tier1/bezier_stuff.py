import bezier
import numpy as np
import matplotlib.pyplot as plt

def bezier_test():
    nodes = np.asfortranarray([
        [0.0, 0.5, 0.5, 1.0],
        [0.0, 0.0, 1.0, 1.0]
    ])
    curve = bezier.Curve(nodes, degree=3)
    s_vals = np.linspace(0.0, 1.0, 10)
    points = curve.evaluate_multi(s_vals)
    print(points)
    # Plot the curve and control points
    ax = curve.plot(10)
    ax.plot(nodes[0, :], nodes[1, :], "o--", color="black")
    ax.plot(points[0, :], points[1, :], "o--", color="red")
    ax.axis("scaled")
    ax.set_title("Quadrotor cubic Bézier curve")
    # Label coordinates of curve points
    for i, point in enumerate(points.T):
        ax.text(
            point[0] + 0.02,
            point[1] - 0.01,
            "({:.3f}, {:.3f})".format(*point),
            ha="left",
            va="top",
            fontsize=10,
        )
    plt.show()
    
def matrix_rotate():
    matrix10 = np.matrix(
        [[0.0, 0.5, 0.5, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32
    )
    print("matrix10")
    print(matrix10)

    x_rot_3d_90 = np.matrix(
        [[1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0]],
        dtype=np.float32
    )

    # Rotate and print
    matrix10_rot = x_rot_3d_90 * matrix10
    print("matrix10_rot")
    print(matrix10_rot)

if __name__ == "__main__":
    bezier_test()
    matrix_rotate()

