# Script to calculate the entropy of a complete file.
import argparse
from library.files import FileObject
from library.plots import ScatterPlot
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import numpy
import time


def main():
    # Capture the running time
    start_time = time.time()

    # Argument parsing
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
                        help="Plot the running entropy values.  Only valid "
                             "if -w is used!"
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
    parser.add_argument("-a", "--anomaly", action='store_true',
                        help="Enable anomaly detection."
                             "", required=False)
    parser.add_argument("-c", "--contamination", type=float, default=0.1,
                        help="Outlier contamination factor."
                             "", required=False)
    parser.add_argument("-l", "--lofneighbors", type=int, default=300,
                        help="Local outlier factor neighbors."
                             "", required=False)

    args = parser.parse_args()

    # Normalize setup...
    if args.nonormalize:
        normalize = False
    else:
        normalize = True

    f = FileObject(args.MalwareFile)
    # Generic Data...
    print("")
    print("**** Summary ****")
    print("Whole file calculations:")
    print("\tFile Name: {0}".format(f.filename))
    print("\tFile Size: {0:,} bytes".format(f.file_size))
    print("\tFile Type: {0}".format(f.filetype))
    print("\tFile Entropy: {0:.6f}".format(f.entropy(normalize)))
    print("\tFile Entropy Calculation Time: {0:.6f} seconds"
          .format(round(time.time() - start_time, 6)))

    # Windows PE Sections...
    if f.parsedfile is not None and f.parsedfile['type'] == 'pefile':
        parsed_time = time.time()
        print("Parsed File As Type: {0}".format(f.parsedfile['type']))
        parsed_entropy = f.parsed_file_entropy(normalize)
        print("\tSection Entropy:")
        for section in parsed_entropy['sections']:
            print("\t\tName: {0}".format(section['name']))
            print("\t\t\tRaw Offset: {0}".format(hex(section['offset'])))
            print("\t\t\tSize: {0}".format(hex(section['length'])))
            print("\t\t\tEntropy: {0:.6f}".format(section['entropy']))
        print("\tParsed File Entropy Calculation Time: {0:6f} seconds"
              .format(round(time.time() - parsed_time, 6)))
    if args.window:
        # Generic data...
        running_start_time = time.time()
        print("Running Window Entropy:")
        print("\tRunning Entropy Window Size (bytes): {0:n}".format(args.window))
        running_entropy = f.running_entropy(int(args.window), normalize)
        n = numpy.array(running_entropy)
        print("\tRunning Entropy Mean: {0:.6f}".format(n.mean()))
        print("\tRunning Entropy Min: {0:.6f}".format(n.min()))
        print("\tRunning Entropy Max: {0:.6f}".format(n.max()))
        print("\tRunning Entropy Calculation Time: {0:.6f} seconds"
              .format(round(time.time() - running_start_time, 6)))

        if args.anomaly:
            # Find anomalies...
            running_start_time = time.time()
            print("Anomalies:")
            anomaly_detector = EllipticEnvelope(contamination=args.contamination)
            # anomaly_detector = LocalOutlierFactor(n_neighbors=args.lofneighbors,
            #                                       n_jobs=10,
            #                                       contamination=args.contamination)
            n_data = n.reshape(-1, 1)
            # anomalies = anomaly_detector.fit_predict(n_data)
            anomaly_detector.fit(n_data)
            anomalies = anomaly_detector.predict(n_data)
            # Fix the data so 1 is an anomaly
            anomalies[anomalies==1] = 0
            anomalies[anomalies==-1] = 1
            anomaly_x_tuple = numpy.where(anomalies==1)
            anomaly_y = anomalies[anomaly_x_tuple]
            anomaly_x = anomaly_x_tuple[0]
            print("\tNumber of anomalies: {0:,}".format(len(anomalies[numpy.where(anomalies == 1)])))
            print("\tContamination: {0:6f}".format(args.contamination))
            print("\tAnomaly Calculation Time: {0:.6f} seconds"
                  .format(round(time.time() - running_start_time, 6)))

        # Windows PE sections...
        if f.parsedfile is not None and f.parsedfile['type'] == 'pefile':
            running_start_time = time.time()
            print("Parsed File Running Window Entropy:")
            print("\tRunning Entropy Window Size (bytes): {0}".format(
                args.window))
            parsed_running_start_time = time.time()
            parsed_running_entropy = \
                f.parsed_file_running_entropy(int(args.window), normalize)
            for section in parsed_running_entropy['sections']:
                n = numpy.array(section['running_entropy'])
                print("\tSection Name: {0}".format(section['name']))
                print("\t\tRunning Entropy Mean: {0:.6f}".format(n.mean()))
                print("\t\tRunning Entropy Min: {0:.6f}".format(n.min()))
                print("\t\tRunning Entropy Max: {0:.6f}".format(n.max()))
            print("\tRunning Entropy Calculation Time: {0:.6f} seconds"
                  .format(round(time.time() - running_start_time, 6)))

        # Plots...
        if args.plotrunningentropy:
            # This will be our HTML output
            html = list()

            # Plot generic data...
            plot_running_start_time = time.time()
            print("Plotting Running Window Entropy...")
            if args.plotrunningentropyskip > 1:
                print("\tSkipping every {0} entropy values in plot..."
                      .format(args.plotrunningentropyskip))
            # Setup the x axis as location information
            x = [i for i in range(0, len(running_entropy))
                 if i % int(args.plotrunningentropyskip) == 0]
            # Setup the y axis values
            y = [round(running_entropy[i], 6)
                 for i in range(0, len(running_entropy))
                 if i % int(args.plotrunningentropyskip) == 0]

            title = ("Malgazer - Running Entropy Window Size {0} Bytes"
                     .format(args.window))
            xtitle = "Byte Location"
            ytitle = "Entropy Value"
            datatitle = ["Entropy"]
            mode = ["lines", "markers"]
            x_vals = [x]
            y_vals = [y]
            if args.anomaly:
                x_vals.append(anomaly_x)
                y_vals.append(anomaly_y)
                datatitle.append('Anomalies (Contamination={0})'.format(args.contamination))

            if args.plotrunningentropyskip > 1:
                xtitle += (" (skip value = {0} bytes)"
                           .format(int(args.plotrunningentropyskip)))
            myplot = ScatterPlot(x=x_vals, datatitle=datatitle, xtitle=xtitle,
                                 y=y_vals, ytitle=ytitle,
                                 plottitle=title, mode=mode)
            html.append(myplot.plot_div())

            # Plot Windows PE sections...
            if f.parsedfile is not None and f.parsedfile['type'] == 'pefile':
                for section in parsed_running_entropy['sections']:
                    # Setup the x axis as location information
                    x = [i for i in range(0, len(section['running_entropy']))
                         if i % int(args.plotrunningentropyskip) == 0]
                    # Setup the y axis values
                    y = [round(section['running_entropy'][i], 6)
                         for i in range(0, len(section['running_entropy']))
                         if i % int(args.plotrunningentropyskip) == 0]

                    title = ("Malgazer - Running Entropy Window Size {0} Bytes"
                             .format(args.window))
                    xtitle = "Byte Location"
                    ytitle = "Entropy Value"
                    datatitle = section['name']
                    if args.plotrunningentropyskip > 1:
                        xtitle += (" (skip value = {0} bytes)"
                                   .format(int(args.plotrunningentropyskip)))
                    myplot = ScatterPlot(x=[x], datatitle=[datatitle],
                                         xtitle=xtitle,
                                         y=[y], ytitle=ytitle,
                                         plottitle=title)
                    html.append(myplot.plot_div())

            with open('malgazer.html', 'w') as m:
                m.write("<HTML><TITLE>Malgazer</TITLE><BODY>")
                for h in html:
                    m.write(h)
                m.write("</BODY></HTML>")
            print("\tPlot Running Window Entropy Time: {0:.6f} seconds"
                  .format(round(time.time() - plot_running_start_time, 6)))

    # Print the running time
    print()
    print("Total running time: {0:.6f} seconds"
          .format(round(time.time() - start_time, 6)))
    print()


if __name__ == "__main__":
    main()
