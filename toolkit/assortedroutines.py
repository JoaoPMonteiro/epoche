import numpy as np
import matplotlib.pyplot as plt


def another_2d_plot(in_point, in_img_width, in_img_height):
    X = in_point[:, 0]
    X = np.append(X, 0)
    X = np.append(X, 0)
    X = np.append(X, in_img_width)
    X = np.append(X, in_img_width)
    Y = (-1) * in_point[:, 1]
    Y = np.append(Y, 0)
    Y = np.append(Y, -in_img_height)
    Y = np.append(Y, 0)
    Y = np.append(Y, -in_img_height)

    nrpts = len(Y)-4
    annotations = [str(ii) if ii < nrpts else '' for ii in range(len(X))]
    plt.figure()
    plt.scatter(X, Y, s=10, color="red")
    plt.xlabel("X")
    plt.ylabel("Y")

    for i, label in enumerate(annotations):
        plt.annotate(label, (X[i], Y[i]))

    plt.show(block=False)


def another_3d_plot(in_point):
    X = in_point[:, 0]
    Y = in_point[:, 1]
    Z = in_point[:, 2]

    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(X, Y, Z, color="green")

    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Y-axis', fontweight='bold')
    ax.set_zlabel('Z-axis', fontweight='bold')

    for i in range(len(X)):
        ax.text(X[i], Y[i], Z[i], '%s' % (str(i)), size=20, zorder=1, color='k')

    plt.show(block=False)
