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

def show3DposePair(realt3d, faket3d, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
                   gt=True, pred=False):  # blue, orange
  """
  Visualize a 3d skeleton pair

  Args
  channels: 96x1 vector. The pose to plot.
  ax: matplotlib 3d axis to draw on
  lcolor: color for left part of the body
  rcolor: color for right part of the body
  add_labels: whether to add coordinate labels
  Returns
  Nothing. Draws on ax.
  """
  #   assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
  realt3d = np.reshape(realt3d, (16, -1))
  faket3d = np.reshape(faket3d, (16, -1))

  I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14])  # start points
  J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])  # end points
  LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  for idx, vals in enumerate([realt3d, faket3d]):
    # Make connection matrix
    for i in np.arange(len(I)):
      x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
      if idx == 0:
        ax.plot(x, z, -y, lw=2, c='k')
      #        ax.plot(x,y, z,  lw=2, c='k')

      elif idx == 1:
        ax.plot(x, z, -y, lw=2, c='r')
      #        ax.plot(x,y, z,  lw=2, c='r')

      else:
        #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

  RADIUS = 1  # space around the subject
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_zlim3d([-RADIUS-yroot, RADIUS-yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")

  # Get rid of the ticks and tick labels
  #  ax.set_xticks([])
  #  ax.set_yticks([])
  #  ax.set_zticks([])
  #
  #  ax.get_xaxis().set_ticklabels([])
  #  ax.get_yaxis().set_ticklabels([])
  #  ax.set_zticklabels([])
  #     ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)
