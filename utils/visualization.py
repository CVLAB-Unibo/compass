import open3d as o3d
from matplotlib import pyplot as plt
import numpy as np

def viz_keypoints(cloud, indices_kp, scale=1.0, color_cloud=[1, 0, 1], color_keypoints=[0, 1, 0]):
    """
    Draw a sphere for each point addressed by index_kp on cloud
    :param cloud: open3D cloud
    :param indices_kp: numpy array with indices of keypoints
    :param scale: scale for keypoints sphere
    :param color_cloud: color for cloud
    :param color_keypoints: color for keypoints on cloud
    :return: list with cloud and keypoints spheres
    """

    cloud.paint_uniform_color(color_cloud)
    viz = [cloud]

    for i in range(indices_kp.shape[0]):

        kp = o3d.geometry.TriangleMesh.create_sphere(scale)
        kp.paint_uniform_color(color_keypoints)

        transform = np.eye(4)
        transform[:, 3] = np.concatenate((cloud.points[indices_kp[i]], [1]), axis=0)

        kp.transform(transform)
        viz.append(kp)

    return viz


def save_clouds_two_views(name_file,
                            clouds,
                            titles,
                            title_figure='',
                            sizes=None,
                            color_map='Dark2',
                            z_axis='y'):
    '''

    :param name_file: output file name
    :param clouds: clouds to visualize
    :param titles: title for each cloud
    :param title_figure: global title of figure
    :param sizes:
    :param color_map: color map to use
    :param z_axis: z_axis of the point cloud
    :param lim_x: limit for x axis
    :param lim_y: limit for y axis
    :param lim_z: limit for z axis
    :return: save a 2D plot containing clouds

    '''

    if sizes is None:
        sizes = [0.5 for i in range(len(clouds))]
    fig = plt.figure(figsize=(len(clouds) * 3, 9))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(clouds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(3, len(clouds), i * len(clouds) + j + 1, projection='3d')
            ax.view_init(elev, azim)

            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=z_axis, c=color, s=size, cmap=color_map, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_xlim((pcd[:, 0].min(), pcd[:, 0].max()))
            ax.set_ylim((pcd[:, 1].min(), pcd[:, 1].max()))
            ax.set_zlim((pcd[:, 2].min(), pcd[:, 2].max()))

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(title_figure)
    fig.savefig(name_file)
    plt.close(fig)

def view_overlapped_clouds(clouds, colors=[], mat_transformations=[]):
    overlapped_clouds = []

    if len(mat_transformations) > 0:
        if len(clouds) == len(colors) and len(colors) == len(mat_transformations):
            for (cloud, mat_transform, color) in zip(clouds, mat_transformations, colors):
                cloud.transform(mat_transform)
                if colors is not None:
                    cloud.paint_uniform_color(color)

                overlapped_clouds.append(cloud)

    else:
        if len(clouds) == len(colors):
            for (cloud, color) in zip(clouds, colors):
                if colors is not None:
                    cloud.paint_uniform_color(color)
                overlapped_clouds.append(cloud)
        else:
            for cloud in clouds:
                overlapped_clouds.append(cloud)

    visualization.draw_geometries(overlapped_clouds)

def save_overlapped_clouds(clouds, colors=[], name_file=""):
    overlapped_clouds = geometry.PointCloud()
    if len(clouds) == len(colors):
        for (cloud, color) in zip(clouds, colors):
            cloud.paint_uniform_color(color)
            overlapped_clouds += cloud
    else:
        for cloud in clouds:
            overlapped_clouds += cloud

    vis = visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    vis.add_geometry(overlapped_clouds)
    # vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(name_file)
    vis.destroy_window()

def get_color_map(x: np.ndarray, name_map: str = "Spectral") -> np.ndarray:
    """Map each scalar value in x to a color using the matplot lib color map.
    :param x: the input data, shape=(N).
    :param name_map: the name of the color map.
    :return: The colors for each sample, shape=(N,3).
    """
    map_color = plt.get_cmap(name_map)
    return map_color(x)[:, :3]