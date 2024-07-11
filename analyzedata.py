import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


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
plt.style.use("seaborn-bright")


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

colors = sns.color_palette(palette="bright", n_colors=3)


df = pd.read_csv("data/runs/object_.csv")
# fig, pos_plot = plt.subplots()
# plothusly(
#     pos_plot, 
#     obj_data["time"], 
#     obj_data["x_pos"], 
#     xtitle="Time [sec]",
#     ytitle="Position [m]",
#     title="Object Null Simulated Position",
#     linestyle='-',
#     datalabel="X Position")\
# plothus(pos_plot, obj_data["time"], obj_data["y_pos"], datalabel="Y Position")
# plothus(pos_plot, obj_data["time"], obj_data["z_pos"], datalabel="Z Position")

meters = mpl.ticker.EngFormatter("m")
newtons = mpl.ticker.EngFormatter("N")
seconds = mpl.ticker.EngFormatter("s")
radians = mpl.ticker.EngFormatter("rad")

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

fig.savefig("data/runs/object__pos.png")

#%%###########################
# Attitude plots
fig, angleplot = plt.subplots()
plothusly(angleplot, df["Time"], df["Pitch"], 
          xtitle="Time [sec]",
          ytitle="Angle from neutral position [rad]",
          datalabel="Pitch", 
          title="Sim Obj Attitude")
plothus(angleplot, df["Time"], df["Yaw"], datalabel="Yaw")
plothus(angleplot, df["Time"], df["Roll"], datalabel="Roll")
angleplot.xaxis.set_major_formatter(seconds)
angleplot.yaxis.set_major_formatter(radians)


fig.savefig("data/runs/object__att.png")
