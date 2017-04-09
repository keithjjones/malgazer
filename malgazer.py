# Script to calculate the entropy of a complete file.
import argparse
from library.files import FileObject
import numpy
import time

try:
    from plotly import __version__
    from plotly.offline import (download_plotlyjs, plot)
    from plotly.graph_objs import (Bar, Scatter, Figure,
    Pie, Layout, Line, Annotations, Annotation)
    from plotly import tools
    from plotly.tools import FigureFactory as FF
except ImportError as e:
    raise e


def main():
    # Capture the running time
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Calculates the entropy of a file.')
    parser.add_argument('MalwareFile',
                        help='The malware file to analyze.')
    parser.add_argument("-n", "--nonormalize", action='store_true',
                        help="Disables entropy normalization."
                             "", required=False)
    parser.add_argument("-w", "--window",
                        help="Window size, in bytes, for running entropy."
                             "", type=int, required=False)
    parser.add_argument("-pre", "--plotrunningentropy", action='store_true',
                        help="Plot the running entropy values."
                             "", required=False)
    parser.add_argument("-preskip", "--plotrunningentropyskip",
                        help="Skip this number of bytes in running entropy "
                             "value plot for compression."
                             "", default=1, type=int, required=False)
    parser.add_argument("-prefilename", "--plotrunningentropyfilename",
                        help="The file name of the output file for a running "
                             "entropy plot (html).",
                        default="malgzer_running_entropy.html",
                        required=False)

    args = parser.parse_args()

    if args.nonormalize:
        normalize = False
    else:
        normalize = True

    f = FileObject(args.MalwareFile)
    print("")
    print("**** Summary ****")
    print("Whole file calculations:")
    print("\tFile Name: {0}".format(f.filename))
    print("\tFile Size: {0:,} bytes".format(f.file_size))
    print("\tFile Type: {0}".format(f.filetype))
    print("\tFile Entropy: {0:.6f}".format(f.entropy(normalize)))
    print("\tFile Entropy Calculation Time: {0:.6f} seconds"
          .format(round(time.time()-start_time, 6)))
    if args.window:
        running_start_time = time.time()
        print("Running Window Entropy:")
        print("\tRunning Entropy Window Size (bytes): {0}".format(args.window))
        running_entropy = f.running_entropy(int(args.window), normalize)
        n = numpy.array(running_entropy)
        print("\tRunning Entropy Mean: {0:.6f}".format(n.mean()))
        print("\tRunning Entropy Min: {0:.6f}".format(n.min()))
        print("\tRunning Entropy Max: {0:.6f}".format(n.max()))
        print("\tRunning Entropy Calculation Time: {0:.6f} seconds"
              .format(round(time.time() - running_start_time, 6)))
        if args.plotrunningentropy:
            plot_running_start_time = time.time()
            print("Plotting Running Window Entropy...")
            if args.plotrunningentropyskip:
                print("\tSkipping every {0} entropy values in plot..."
                      .format(args.plotrunningentropyskip))
            # Setup the x axis as location information
            x = [i for i in range(0, len(running_entropy))
                 if i % int(args.plotrunningentropyskip) == 0]
            # Setup the y axis values
            y = [round(running_entropy[i], 6)
                 for i in range(0, len(running_entropy))
                 if i % int(args.plotrunningentropyskip) == 0]

            # Start with empty output
            output = []
            title = "Running Window Entropy"

            # Add current scatter plot
            output.append(Scatter(name="Running Entropy Window Size {0} Bytes"
                                  .format(args.window),
                                  x=x,
                                  y=y,
                                  hoverinfo="x+y"
                                  ))

            # Setup the plot...
            xaxis_title = "Byte Location"
            if args.plotrunningentropyskip > 1:
                xaxis_title += (" (skip value = {0} bytes)"
                                .format(int(args.plotrunningentropyskip)))
            plotlayout = Layout(showlegend=True, title=title,
                                xaxis=dict(title=xaxis_title),
                                yaxis=dict(title="Entropy"))
            plotfigure = Figure(data=output, layout=plotlayout)

            # Plot without the plotly annoying link...
            plot(plotfigure, show_link=False,
                 filename=args.plotrunningentropyfilename,
                 auto_open=True)
            print("\tPlot Running Window Entropy Time: {0:.6f} seconds"
                  .format(round(time.time()-plot_running_start_time, 6)))

    # Print the running time
    print()
    print("Total running time: {0:.6f} seconds"
          .format(round(time.time()-start_time, 6)))
    print()

if __name__ == "__main__":
    main()
