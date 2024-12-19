import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from collections import defaultdict
import re
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import seaborn as sns
object_name = "drone"
def plothusly(ax, x, y, *, xtitle='', ytitle='',
              datalabel='', title='', linestyle='-',
              marker=''):
    """
    A little function to make graphing less of a pain.
    Creates a plot with titles and axis labels.
    Adds a new line to a blank figure and labels it.

    Parameters
    ----------
    ax : The graph object
    x : X axis data
    y : Y axis data
    xtitle : Optional x axis data title. The default is ''.
    ytitle : Optional y axis data title. The default is ''.
    datalabel : Optional label for data. The default is ''.
    title : Graph Title. The default is ''.

    Returns
    -------
    out : Resultant graph.

    """
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_title(title)
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle=linestyle,
                  marker = marker)
    ax.grid(True)
    ax.legend(loc='best')
    return out


def plothus(ax, x, y, *, datalabel='', linestyle = '-',
            marker = ''):
    """
    A little function to make graphing less of a pain

    Adds a new line to a blank figure and labels it
    """
    out = ax.plot(x, y, zorder=1, label=datalabel, linestyle = linestyle,
                  marker = marker)
    ax.legend(loc='best')
    return out

plt.style.use("default")
# plt.style.use("seaborn-bright")


params={#FONT SIZES
    # 'axes.labelsize':45,#Axis Labels
    # 'axes.titlesize':35,#Title
    # 'font.size':28,#Textbox
    # 'xtick.labelsize':30,#Axis tick labels
    # 'ytick.labelsize':30,#Axis tick labels
    # 'legend.fontsize':30,#Legend font size
    'font.family':'sans-serif',
    'font.fantasy':'xkcd',
    'font.sans-serif':'Helvetica',
    'font.monospace':'Courier',
    #AXIS PROPERTIES
    'axes.titlepad':2*6.0,#title spacing from axis
    'axes.grid':True,#grid on plot
    'figure.figsize':(16, 10),#square plots
    # 'savefig.bbox':'tight',#reduce whitespace in saved figures#LEGEND PROPERTIES
    'legend.framealpha':0.5,
    'legend.fancybox':True,
    'legend.frameon':True,
    'legend.numpoints':1,
    'legend.scatterpoints':1,
    'legend.borderpad':0.1,
    'legend.borderaxespad':0.1,
    'legend.handletextpad':0.2,
    'legend.handlelength':1.0,
    'legend.labelspacing':0,}
mpl.rcParams.update(params)

# colors = sns.color_palette(palette="bright", n_colors=3)

# object_name = "ISS_0"
# df = pd.read_csv(f"data/runs/object_{object_name}.csv")
# df = pd.read_csv("data/runs/test_constellation/object_0_ISS_0_")

# filelist = glob.glob("data/runs/test_constellation/object_0_ISS_0_" + "*.csv")

# li = []

# for filename in filelist:

#     df = pd.read_csv(filename, index_col=None, header=0, usecols=[0,5])

#     li.append(df)

# df = pd.concat(li, axis=0, ignore_index=True)

# REVISIT ME:

@dataclass
class DataStorage:
    id: str
    data: pd.DataFrame

def combine_csv_files(dir):
    """
    Combines CSV files for each object into a single CSV file.

    Args:
    input_directory (str): Path to the directory containing the input CSV files.
    output_directory (str): Path to the directory where the combined CSV files will be saved.

    Returns:
    None
    """
    # Ensure output directory exists
    input_directory = dir
    output_directory = dir
     # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Dictionary to store DataFrames for each object
    object_data = defaultdict(list)

    # Regular expression to extract object_id from filename
    pattern = r'(.+?)_\d+\.csv$'

    to_delete =[]

    # Iterate over all CSV files in the input directory
    for filename in sorted(os.listdir(input_directory)):
        # print(filename)
        if filename.endswith('.csv'):
            match = re.match(pattern, filename)
            if match:
                object_id = match.group(1)
                file_path = os.path.join(input_directory, filename)
                df = pd.read_csv(file_path)
                object_data[object_id].append(df)
                to_delete.append(file_path)
    combined_list = []
    # Combine and save DataFrames for each object
    for object_id, dataframes in object_data.items():
        combined_df = pd.concat(dataframes, ignore_index=True)
        output_file = os.path.join(output_directory, f"{object_id}_combined.csv")
        combined_df.to_csv(output_file, index=False)
        combined_df = combined_df.sort_values("Time", ascending=True)
        # print(f"Combined CSV for {object_id} saved to {output_file}")
        combined_list.append(DataStorage(object_id, combined_df))

    # [os.remove(old_file) for old_file in to_delete]

    return combined_list


def plot_trajectories(combined_data):
    """
    Plot 3D trajectories of all objects using Plotly.

    Args:
    combined_data (list): List of tuples (object_id, dataframe) as returned by combine_csv_files.

    Returns:
    plotly.graph_objects.Figure: The Plotly figure object.
    """
    fig = go.Figure()

    for object in combined_data:
        object_id = object.id
        df = object.data
        fig.add_trace(go.Scatter3d(
            x=df['X Position'], y=df['Y Position'], z=df['Z Position'],
            mode='lines',
            name=object_id
        ))

    fig.update_layout(
        title="3D Trajectories of Objects",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        legend_title="Objects"
    )

    return fig

def animate_trajectories(combined_data):
    """
    Create an animated 3D plot of trajectories using Plotly.

    Args:
    combined_data (list): List of tuples (object_id, dataframe) as returned by combine_csv_files.

    Returns:
    plotly.graph_objects.Figure: The Plotly figure object with animation.
    """
    fig = go.Figure()

    # Determine the maximum number of frames
    max_frames = max(len(object.data) for object in combined_data)

    for object in combined_data:
        
        fig.add_trace(go.Scatter3d(
            x=object.data['X Position'][:1], y=object.data['Y Position'][:1], z=object.data['Z Position'][:1],
            mode='lines',
            name=object.id
        ))

    # Create frames for animation
    frames = [go.Frame(data=[go.Scatter3d(
        x=object.data['X Position'][:k+1], y=object.data['Y Position'][:k+1], z=object.data['Z Position'][:k+1]
    ) for object in combined_data]) 
              for k in range(1, max_frames)]

    fig.frames = frames

    fig.update_layout(
        title="Animated 3D Trajectories of Objects",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        legend_title="Objects",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 50, "redraw": True},
                                       "fromcurrent": True}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}])]
        )]
    )

    return fig

data = combine_csv_files(path:="data/runs/test_drone/")
# print(data[0].data.head())
# data = [pd.read_csv("data/runs/test_constellation/"+file) for file in os.listdir("data/runs/test_constellation/")]
# print(data[0].data.head())
# df = data[1]

# Generate static plot
static_fig = plot_trajectories(data)
# static_fig.show()

# Generate animated plot
# animated_fig = animate_trajectories(data)
# animated_fig.show()

# static_fig.write_html(path+"static_trajectories.html")
# animated_fig.write_html(path+"animated_trajectories.html")



# fig, pos_plot = plt.subplots()
# plothusly(
#     pos_plot, 
#     obj_data["time"], 
#     obj_data["x_pos"], 
#     xtitle="Time [sec]",
#     ytitle="Position [m]",
#     title="Object Null Simulated Position",
#     linestyle='-',
#     datalabel="X Position")
# plothus(pos_plot, obj_data["time"], obj_data["y_pos"], datalabel="Y Position")
# plothus(pos_plot, obj_data["time"], obj_data["z_pos"], datalabel="Z Position")

meters = mpl.ticker.EngFormatter("m")
newtons = mpl.ticker.EngFormatter("N")
seconds = mpl.ticker.EngFormatter("s")
radians = mpl.ticker.EngFormatter("rad")

df = data[0].data
# Position plot
fig, zplot = plt.subplots()
plothusly(zplot, df["Time"], df["Z Position"], 
          xtitle="Time [sec]",
          ytitle="Position [m]]", 
          datalabel="Z Position", 
          title="Sim Obj Position")
plothus(zplot, df["Time"], df["Y Position"], datalabel="Y Position")
plothus(zplot, df["Time"], df["X Position"], datalabel="X Position")

zplot.xaxis.set_major_formatter(seconds)
zplot.yaxis.set_major_formatter(meters)

fig.savefig(f"data/runs/test_drone/object_{object_name}_pos.png")

# #%%###########################
# # Attitude plots
# fig, angleplot = plt.subplots()
# plothusly(angleplot, df["Time"], df["Pitch"], 
#           xtitle="Time [sec]",
#           ytitle="Angle from neutral position [rad]",
#           datalabel="Pitch", 
#           title="Sim Obj Attitude")
# plothus(angleplot, df["Time"], df["Yaw"], datalabel="Yaw")
# plothus(angleplot, df["Time"], df["Roll"], datalabel="Roll")
# angleplot.xaxis.set_major_formatter(seconds)
# angleplot.yaxis.set_major_formatter(radians)


# fig.savefig(f"data/runs/object_{object_name}_att.png")

# #%%###########################

# fig, threedplot = plt.subplots()
# fig.add_subplot(projection='3d')

# # Prepare arrays x, y, z
# theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
# z = np.linspace(-2, 2, 100)
# r = z**2 + 1
# x = r * np.sin(theta)
# y = r * np.cos(theta)

# threedplot.plot(x, y, z, label='parametric curve')
# threedplot.legend()

# fig.savefig(f"data/runs/object_{object_name}_3d_plot.png")