import argparse

import matplotlib.font_manager as fm  # Collect all the font names available to matplotlib
import matplotlib.pyplot as plt
import numpy as np

import artistic as art

font_names = [f.name for f in fm.fontManager.ttflist]

# mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["figure.autolayout"] = True

#################################


parser = argparse.ArgumentParser()

parser.add_argument("--directory", type=str, default="./")
parser.add_argument("--filename", type=str, default="data.npy")
parser.add_argument("--plotname", type=str, default="plot")
args = parser.parse_args()

data2plot = np.load(args.directory + args.filename).astype(np.float32)
data2plot = data2plot[0]

canvas = art.canvasHolder("MassifCentral", 256, 256)
Datamax = data2plot.max(axis=(0, -1, -2))
print(Datamax)
Datamin = data2plot.min(axis=(0, -1, -2))
print(Datamin)
var_names = [("u", "m/s"), ("v", "m/s"), ("t2m", "K")]
# data2plot0 = data2plot[(0,1,2,4),:,:,:]
canvas.plot_data_normal(
    data2plot,
    var_names,
    args.directory,
    f"{args.plotname}.pdf",
    contrast=True,
    cvalues=(Datamin, Datamax),
)

# canvas.plot_data_normal(data2plot[3:4], [('t2m', 'K'),('t2m', 'K'),('t2m', 'K')],
#  args.directory, f"{args.plotname}_diff.pdf", contrast=True,
#                    cvalues=((-4.0,-4.0,-3.0), (4.0,4.0,3.0)))

var_names = [("ff", "m/s"), ("t2m", "K")]
canvas.plot_data_ff_t2m(
    data2plot,
    var_names,
    args.directory,
    f"{args.plotname}_fft2m.pdf",
    contrast=False,
)
