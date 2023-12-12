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
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0]],
        dtype=np.float32
    )
    np.set_printoptions(precision=2)
    print("matrix10")
    print(matrix10)

    x_rot_3d_45 = np.matrix(
        [[1.0, 0.0, 0.0],
        [0.0, np.cos(np.deg2rad(45)), -np.sin(np.deg2rad(45))],
        [0.0, np.sin(np.deg2rad(45)), np.cos(np.deg2rad(45))]],
        dtype=np.float32
    )

    # Rotate and print
    matrix11 = x_rot_3d_45 * matrix10
    print("matrix11")
    print(matrix11)

    # Rotate and print
    matrix12 = x_rot_3d_45 * matrix11
    print("matrix12")
    print(matrix12)

    # Rotate and print
    matrix13 = x_rot_3d_45 * matrix12
    print("matrix13")
    print(matrix13)

    # Rotate and print
    matrix14 = x_rot_3d_45 * matrix13
    print("matrix14")
    print(matrix14)

    # Rotate and print
    matrix15 = x_rot_3d_45 * matrix14
    print("matrix15")
    print(matrix15)

    # Rotate and print
    matrix16 = x_rot_3d_45 * matrix15
    print("matrix16")
    print(matrix16)

    # Rotate and print
    matrix17 = x_rot_3d_45 * matrix16
    print("matrix17")
    print(matrix17)

def stay_still():
    nodes_0 = np.asfortranarray([
        [0.0, 0.0],
        [0.0, 0.0]
    ])
    curve_0 = bezier.Curve(nodes_0, degree=1)
    s_vals = np.linspace(0.0, 1.0, 10)
    points = curve_0.evaluate_multi(s_vals)
    print(points)
    # Plot the curve and control points
    ax = curve_0.plot(10)
    ax.plot(nodes_0[0, :], nodes_0[1, :], "o--", color="black")
    ax.plot(points[0, :], points[1, :], "o--", color="red")
    ax.axis("scaled")
    ax.set_title("Stay still curve")
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

if __name__ == "__main__":
    bezier_test()
    matrix_rotate()
    # stay_still()

